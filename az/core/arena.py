"""Head-to-head evaluation between two checkpoints.

This is your progress meter, and it is not optional. Training loss goes down whether or
not the engine is getting stronger -- the only trustworthy signal is that generation N
beats generation N-10 over enough games. Without this you cannot tell real progress from
the net getting better at predicting its own noise.

Games are played with root noise disabled and greedy move selection after a few
randomised opening plies (otherwise every game between two deterministic players is
identical and the result is meaningless).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from az.core.game import Game
from az.core.mcts import BatchedMCTS, Evaluator, MCTSConfig


@dataclass
class ArenaResult:
    wins: int  # for the first (candidate) player
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        """Points per game in [0, 1]."""
        if self.games == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def elo_diff(self) -> float:
        """Estimated Elo difference, clamped so a clean sweep is finite."""
        score = min(max(self.score, 1e-3), 1 - 1e-3)
        return -400.0 * math.log10(1.0 / score - 1.0)

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "score": round(self.score, 4),
            "elo_diff": round(self.elo_diff, 1),
        }

    def __str__(self) -> str:
        return (
            f"+{self.wins} ={self.draws} -{self.losses} "
            f"(score {self.score:.1%}, {self.elo_diff:+.0f} Elo)"
        )


def play_match(
    game: Game,
    candidate: Evaluator,
    baseline: Evaluator,
    num_games: int = 40,
    num_simulations: int = 200,
    parallel_games: int = 16,
    random_opening_plies: int = 6,
    seed: Optional[int] = None,
) -> ArenaResult:
    """Play `num_games` with colours alternating evenly.

    Both engines search the same batch of positions in lockstep, so a match costs about
    the same as generating the same number of self-play games.
    """
    rng = np.random.default_rng(seed)
    mcts_config = MCTSConfig(num_simulations=num_simulations)
    searchers = {
        0: BatchedMCTS(game, candidate, mcts_config),
        1: BatchedMCTS(game, baseline, mcts_config),
    }

    wins = losses = draws = 0
    remaining = num_games

    while remaining > 0:
        width = min(parallel_games, remaining)
        # Half the slots start with the candidate as first player.
        candidate_is_first = [i % 2 == 0 for i in range(width)]
        states = [game.initial_state() for _ in range(width)]
        plies = [0] * width
        active = list(range(width))

        while active:
            # Group by which engine is on turn, so each forward pass uses one network.
            groups: dict[int, list[int]] = {0: [], 1: []}
            for slot in active:
                mover_is_first = plies[slot] % 2 == 0
                engine = (
                    0 if mover_is_first == candidate_is_first[slot] else 1
                )
                groups[engine].append(slot)

            for engine, slots in groups.items():
                if not slots:
                    continue
                results = searchers[engine].search(
                    [states[s] for s in slots], add_root_noise=False, rng=rng
                )
                for slot, result in zip(slots, results):
                    visits = result.visit_counts
                    if visits.sum() <= 0:
                        continue
                    # Randomise the opening so the match is not one game repeated.
                    if plies[slot] < random_opening_plies:
                        probs = visits / visits.sum()
                        action = int(rng.choice(len(visits), p=probs))
                    else:
                        action = int(visits.argmax())
                    states[slot] = game.next_state(states[slot], action)
                    plies[slot] += 1

            still_active = []
            for slot in active:
                value = game.terminal_value(states[slot])
                if value is None:
                    still_active.append(slot)
                    continue

                # `value` is from the perspective of the player to move at the
                # terminal position, which is the player who did NOT just move.
                mover_is_first = plies[slot] % 2 == 0
                first_player_value = value if mover_is_first else -value
                candidate_value = (
                    first_player_value
                    if candidate_is_first[slot]
                    else -first_player_value
                )

                if candidate_value > 0:
                    wins += 1
                elif candidate_value < 0:
                    losses += 1
                else:
                    draws += 1

            active = still_active

        remaining -= width

    return ArenaResult(wins=wins, losses=losses, draws=draws)
