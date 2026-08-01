"""Diagnose what a checkpoint actually understands, without spending cloud money.

    python tactics.py --run runs/chess_beam
    python tactics.py --run runs/chess_beam --simulations 800

Two tests, answering different questions:

1. MATERIAL SENSE queries the value head directly, with no search at all. If the network
   cannot tell "up a queen" from "level", nothing search does will save it -- search only
   amplifies the evaluation it is given. This is the test that explained a real failure:
   an engine that captured a free queen in its search lines and still evaluated the
   position at 0.00.

2. TACTICS asks whether search finds forced wins. Terminal results (checkmate) are found
   by the game rules regardless of the value head, so a checkpoint can score well here
   and still hang pieces every move. Read it alongside test 1, never alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np
import torch

from az.core.checkpoint import CheckpointManager, load_net_from_checkpoint
from az.core.game import make_game
from az.core.mcts import BatchedMCTS, MCTSConfig, torch_evaluator

# Simple endgames with a known material edge for the side to move.
# Value should rise monotonically with the advantage if the network understands material.
MATERIAL_POSITIONS = [
    ("down a queen", "3qk3/8/8/8/8/8/8/4K3 w - - 0 1", -9),
    ("down a rook", "3rk3/8/8/8/8/8/8/4K3 w - - 0 1", -5),
    ("down a knight", "3nk3/8/8/8/8/8/8/4K3 w - - 0 1", -3),
    ("level", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
    ("up a knight", "4k3/8/8/8/8/8/8/3NK3 w - - 0 1", 3),
    ("up a rook", "4k3/8/8/8/8/8/8/3RK3 w - - 0 1", 5),
    ("up a queen", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 9),
]

# Each test is judged by OUTCOME, not by a specific move: "capture the piece on this
# square", or "deliver mate". Hardcoding one winning move produced false failures --
# the engine answered Kxe2 where the list only allowed Rxe2, and both win the rook.
TACTICS = [
    ("capture free queen", "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "capture", "d5"),
    ("capture free rook", "4k3/8/8/8/8/8/3r4/3QK3 w - - 0 1", "capture", "d2"),
    ("mate in 1", "6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1", "mate", None),
    ("win the rook", "4k3/8/8/8/8/8/4r3/4RK2 w - - 0 1", "capture", "e2"),
    ("win the bishop", "4k3/8/8/8/8/8/5b2/4K2R w - - 0 1", "capture", "f2"),
]


def material_sense(game, net, device) -> float:
    """Correlate the value head's output with actual material. Returns Pearson r."""
    print("material sense (value head only, no search)")
    print(f"  {'position':22s} {'material':>9s} {'value':>8s}")

    advantages, values = [], []
    for label, fen, advantage in MATERIAL_POSITIONS:
        state = game.initial_state(fen)
        observation = game.encode(state)[None].astype(np.float32)
        mask = game.legal_actions(state)[None]
        _, value = torch_evaluator(net, device)(observation, mask)
        value = float(value[0])
        advantages.append(advantage)
        values.append(value)
        print(f"  {label:22s} {advantage:+9d} {value:+8.3f}")

    if len(set(values)) == 1:
        print("\n  every position scored identically: the value head is material-blind")
        return 0.0

    correlation = float(np.corrcoef(advantages, values)[0, 1])
    spread = max(values) - min(values)
    print(f"\n  correlation with material: {correlation:+.3f}")
    print(f"  value spread across +/- a queen: {spread:.3f}")

    # Correlation says it ranks material correctly; spread says how much it thinks
    # material MATTERS. A network can score 0.9 correlation while compressing every
    # position to within 0.07 of a draw -- which is materially blind in practice,
    # because search has nothing to steer on.
    if correlation < 0.5:
        print("  => no material understanding at all")
    elif spread < 0.15:
        print("  => ranks material but treats everything as a draw: effectively blind.")
        print("     Check draw labelling (see adjudicate_material_at).")
    elif spread < 0.5:
        print("  => learning material; the signal is real but still compressed")
    else:
        print("  => the value head understands material")
    return correlation


def tactics(game, net, device, simulations: int) -> int:
    print(f"\ntactics ({simulations} simulations)")
    evaluator = torch_evaluator(net, device)
    correct = 0

    for label, fen, kind, target in TACTICS:
        state = game.initial_state(fen)
        mcts = BatchedMCTS(game, evaluator, MCTSConfig(num_simulations=simulations))
        result = mcts.search([state], add_root_noise=False)[0]
        move = game.action_to_move(state, result.best_action)
        played = move.uci() if move else "none"

        if move is None:
            hit = False
        elif kind == "capture":
            hit = chess.square_name(move.to_square) == target
        else:
            board = state.board.copy()
            board.push(move)
            hit = board.is_checkmate()

        correct += hit
        goal = f"take {target}" if kind == "capture" else "mate"
        print(
            f"  {label:24s} played {played:6s} goal {goal:9s} "
            f"{'OK  ' if hit else 'MISS'} eval {result.root_value:+.2f}"
        )

    print(f"  => {correct}/{len(TACTICS)}")
    return correct


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a checkpoint")
    parser.add_argument("--run", type=Path, default=Path("runs/chess_beam"))
    parser.add_argument("--generation", type=int)
    parser.add_argument("--simulations", type=int, default=400)
    args = parser.parse_args()

    manager = CheckpointManager(args.run)
    checkpoint = manager.load(args.generation)
    if checkpoint is None:
        print(f"no checkpoint in {args.run}")
        return 1

    device = torch.device("cpu")
    game = make_game(checkpoint.game_name, **checkpoint.game_kwargs)
    net = load_net_from_checkpoint(checkpoint, device)

    print(f"generation {checkpoint.run_state.generation}, "
          f"{checkpoint.run_state.total_games} games\n")

    material_sense(game, net, device)
    tactics(game, net, device, args.simulations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
