"""Batched PUCT Monte-Carlo Tree Search.

The one design decision that matters for throughput: this class searches N *independent*
games simultaneously, stepping every tree one simulation at a time in lockstep. Each
round collects at most one leaf per game and evaluates all of them in a single network
forward pass, so the GPU sees batches of N instead of batches of 1.

That is worth roughly 20-50x over the obvious one-game-at-a-time implementation, and it
is the difference between a training session that makes visible progress and one that
does not. Because the trees are independent, no virtual loss is needed.

There are no rollouts. Leaf values come from the network's value head, or from the exact
game result at terminal nodes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from az.core.game import Game

# (observations, legal_masks) -> (policy_probs, values)
Evaluator = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct_init: float = 1.25
    c_puct_base: float = 19652.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    fpu_reduction: float = 0.0


class Node:
    """A single search-tree node.

    Deliberately does NOT store the game state. States are reconstructed by replaying
    actions from the root during descent, which keeps memory flat -- storing an
    8-ply-history chess position at every one of 800 nodes across 64 parallel games
    would not fit comfortably in RAM.
    """

    __slots__ = ("prior", "visit_count", "value_sum", "children", "terminal_value")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}
        self.terminal_value: Optional[float] = None

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        """Mean value from the perspective of the player to move at this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class SearchResult:
    visit_counts: np.ndarray  # (action_size,) float32
    root_value: float
    best_action: int
    principal_variation: list[int]
    nodes: int = 0


class BatchedMCTS:
    def __init__(self, game: Game, evaluator: Evaluator, config: MCTSConfig) -> None:
        self.game = game
        self.evaluator = evaluator
        self.config = config

    # --- public API ---------------------------------------------------------

    def search(
        self,
        states: Sequence,
        add_root_noise: bool = True,
        rng: Optional[np.random.Generator] = None,
        deadline: Optional[float] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> list[SearchResult]:
        """Run simulations on every state in `states`.

        Stops at `config.num_simulations`, or early if `deadline` (an absolute
        time.monotonic() value) passes or `should_stop()` returns True. The early-exit
        paths are what UCI time control and the GUI's `stop` command hang off.
        """
        rng = rng or np.random.default_rng()
        roots = [Node(prior=1.0) for _ in states]

        self._expand_batch(roots, list(states))
        if add_root_noise:
            for root in roots:
                self._add_dirichlet_noise(root, rng)

        for simulation in range(self.config.num_simulations):
            # Check every few rounds rather than every round: for large batches the
            # clock call is negligible, but for batch-of-1 GUI search it is not.
            #
            # `simulation > 0` matters: checking at simulation 0 lets an already-set
            # stop flag (a GUI that sends `go` and `stop` back to back, or an expired
            # clock) abort before a single simulation runs, leaving the root with no
            # visits and no move to report.
            if simulation > 0 and simulation % 8 == 0:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if should_stop is not None and should_stop():
                    break
            self._simulate_round(roots, states)

        return [self._collect(root) for root in roots]

    # --- one lockstep simulation across every tree --------------------------

    def _simulate_round(self, roots: list[Node], states: Sequence) -> None:
        pending_paths: list[list[Node]] = []
        pending_nodes: list[Node] = []
        pending_states: list = []

        for root, root_state in zip(roots, states):
            path = [root]
            node = root
            state = self.game.copy(root_state)

            while node.is_expanded:
                action, node = self._select_child(node)
                state = self.game.next_state(state, action)
                path.append(node)

            # A leaf: either a game end (exact value, no network needed) or a
            # position to hand to the network.
            if node.terminal_value is None:
                node.terminal_value = self.game.terminal_value(state)

            if node.terminal_value is not None:
                self._backup(path, node.terminal_value)
            else:
                pending_paths.append(path)
                pending_nodes.append(node)
                pending_states.append(state)

        if not pending_nodes:
            return

        values = self._expand_batch(pending_nodes, pending_states)
        for path, value in zip(pending_paths, values):
            self._backup(path, float(value))

    def _expand_batch(self, nodes: list[Node], states: list) -> np.ndarray:
        """Evaluate `states` in one forward pass and attach children to `nodes`."""
        observations = np.stack([self.game.encode(s) for s in states]).astype(np.float32)
        masks = np.stack([self.game.legal_actions(s) for s in states])

        priors, values = self.evaluator(observations, masks)

        for node, prior_row, mask_row in zip(nodes, priors, masks):
            legal = np.flatnonzero(mask_row)
            if legal.size == 0:
                continue
            probs = prior_row[legal]
            total = float(probs.sum())
            # A freshly initialised network can put ~0 mass on every legal move once
            # illegal moves are masked out; fall back to uniform so search still works.
            if not np.isfinite(total) or total <= 1e-8:
                probs = np.full(legal.size, 1.0 / legal.size, dtype=np.float32)
            else:
                probs = probs / total
            node.children = {
                int(action): Node(prior=float(p)) for action, p in zip(legal, probs)
            }

        return values

    # --- selection and backup ----------------------------------------------

    def _select_child(self, node: Node) -> tuple[int, Node]:
        cfg = self.config
        # AlphaZero's c_puct grows slowly with visit count, shifting the balance from
        # prior-driven exploration toward value-driven exploitation as the tree fills.
        c_puct = (
            math.log((1 + node.visit_count + cfg.c_puct_base) / cfg.c_puct_base)
            + cfg.c_puct_init
        )
        sqrt_total = math.sqrt(max(node.visit_count, 1))
        parent_value = node.value

        best_score = -float("inf")
        best_action = -1
        best_child: Optional[Node] = None

        for action, child in node.children.items():
            if child.visit_count > 0:
                # Negated: the child's value is from the *child's* mover's view, and
                # every score here must be in THIS node's mover's view.
                q = -child.value
            elif cfg.fpu_reduction:
                # First-play urgency: value an unvisited child at the parent's own
                # estimate, minus a penalty. `parent_value` is already in this node's
                # mover's frame, so it is used directly and NOT negated.
                q = parent_value - cfg.fpu_reduction
            else:
                q = 0.0  # the paper's choice: unvisited children look neutral
            u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None, "expanded node with no children"
        return best_action, best_child

    @staticmethod
    def _backup(path: list[Node], value: float) -> None:
        """Propagate `value` (from the leaf mover's perspective) up to the root,
        flipping sign at every ply because the players alternate."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _add_dirichlet_noise(self, node: Node, rng: np.random.Generator) -> None:
        if not node.children or self.config.dirichlet_epsilon <= 0:
            return
        actions = list(node.children)
        noise = rng.dirichlet([self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon
        for action, n in zip(actions, noise):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    # --- result extraction --------------------------------------------------

    def _collect(self, root: Node) -> SearchResult:
        visits = np.zeros(self.game.action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if visits.sum() > 0:
            best_action = int(visits.argmax())
        elif root.children:
            # Search was cut off before any simulation completed. The root is still
            # expanded, so the network's policy is strictly better than giving up --
            # returning -1 here forces the caller into an arbitrary-move fallback.
            best_action = max(root.children.items(), key=lambda kv: kv[1].prior)[0]
        else:
            best_action = -1  # genuinely no legal moves

        pv: list[int] = []
        node = root
        while node.children:
            action, node = max(
                node.children.items(), key=lambda kv: kv[1].visit_count
            )
            if node.visit_count == 0:
                break
            pv.append(action)

        return SearchResult(
            visit_counts=visits,
            root_value=root.value,
            best_action=best_action,
            principal_variation=pv,
            nodes=root.visit_count,
        )


def torch_evaluator(net, device, use_amp: bool = False) -> Evaluator:
    """Adapt a torch AlphaZeroNet into the numpy-in/numpy-out Evaluator protocol."""
    import torch

    def evaluate(observations: np.ndarray, masks: np.ndarray):
        obs = torch.from_numpy(observations).to(device, non_blocking=True)
        msk = torch.from_numpy(masks).to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                probs, values = net.predict(obs, msk)
        else:
            probs, values = net.predict(obs, msk)
        return probs.float().cpu().numpy(), values.float().cpu().numpy()

    return evaluate


def uniform_evaluator(game: Game) -> Evaluator:
    """A network-free evaluator: uniform priors, zero values.

    Useful for testing MCTS mechanics in isolation from any learning.
    """

    def evaluate(observations: np.ndarray, masks: np.ndarray):
        counts = masks.sum(axis=1, keepdims=True).clip(min=1)
        probs = masks.astype(np.float32) / counts
        values = np.zeros(len(observations), dtype=np.float32)
        return probs, values

    return evaluate
