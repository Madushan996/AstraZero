"""Command line for training, inspecting and evaluating a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from az.core.arena import play_match
from az.core.checkpoint import CheckpointManager, load_net_from_checkpoint
from az.core.game import make_game
from az.core.mcts import torch_evaluator
from az.core.pipeline import PipelineConfig, TrainingSession, default_config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run", type=Path, required=True, help="run directory (created if absent)"
    )


def cmd_train(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    existing = (run_dir / "config.json").exists()

    config = None
    if not existing:
        config = default_config(args.game, profile=args.profile)
        if args.simulations:
            config.selfplay["num_simulations"] = args.simulations
        if args.parallel:
            config.selfplay["parallel_games"] = args.parallel
    elif args.profile_override:
        config = PipelineConfig.from_dict(
            json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        )
        if args.simulations:
            config.selfplay["num_simulations"] = args.simulations
        if args.parallel:
            config.selfplay["parallel_games"] = args.parallel

    session = TrainingSession(run_dir, config=config, device=args.device)
    session.run_session(
        minutes=args.minutes,
        generations=args.generations,
        selfplay_fraction=args.selfplay_fraction,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manager = CheckpointManager(args.run)
    summary = manager.summary()
    print(json.dumps(summary, indent=2))

    history = manager.read_history()
    if not history:
        print("\nno generations recorded yet")
        return 0

    print(f"\nlast {min(args.last, len(history))} generations:")
    header = f"{'gen':>5} {'games':>6} {'plies':>6} {'loss':>8} {'policy':>8} {'value':>8}"
    print(header)
    print("-" * len(header))
    for entry in history[-args.last :]:
        selfplay = entry.get("selfplay", {})
        train = entry.get("train", {})
        print(
            f"{entry.get('generation', 0):>5} "
            f"{selfplay.get('games', 0):>6} "
            f"{selfplay.get('avg_plies', 0):>6.0f} "
            f"{train.get('loss', 0):>8.4f} "
            f"{train.get('policy_loss', 0):>8.4f} "
            f"{train.get('value_loss', 0):>8.4f}"
        )

    evaluations = [e for e in history if "arena" in e]
    if evaluations:
        print("\nhead-to-head evaluations:")
        for entry in evaluations[-10:]:
            arena = entry["arena"]
            print(
                f"  gen {arena['candidate']} vs gen {arena['baseline']}: "
                f"+{arena['wins']} ={arena['draws']} -{arena['losses']} "
                f"({arena['elo_diff']:+.0f} Elo)"
            )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    manager = CheckpointManager(args.run)
    generations = manager.available_generations()
    if len(generations) < 2:
        print(f"need at least 2 checkpoints to compare; have {len(generations)}")
        return 1

    candidate_gen = args.candidate if args.candidate is not None else generations[-1]
    if args.baseline is not None:
        baseline_gen = args.baseline
    else:
        older = [g for g in generations if g <= candidate_gen - args.gap]
        baseline_gen = older[-1] if older else generations[0]

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    candidate_ckpt = manager.load(candidate_gen, map_location=str(device))
    baseline_ckpt = manager.load(baseline_gen, map_location=str(device))
    if candidate_ckpt is None or baseline_ckpt is None:
        print("could not load one of the requested checkpoints")
        return 1

    game = make_game(candidate_ckpt.game_name, **candidate_ckpt.game_kwargs)
    candidate_net = load_net_from_checkpoint(candidate_ckpt, device)
    baseline_net = load_net_from_checkpoint(baseline_ckpt, device)

    print(
        f"gen {candidate_gen} vs gen {baseline_gen}: "
        f"{args.games} games at {args.simulations} sims..."
    )
    result = play_match(
        game,
        torch_evaluator(candidate_net, device),
        torch_evaluator(baseline_net, device),
        num_games=args.games,
        num_simulations=args.simulations,
        parallel_games=args.parallel,
        seed=args.seed,
    )
    print(f"gen {candidate_gen}: {result}")

    manager.append_history(
        {
            "generation": candidate_gen,
            "arena": {
                "candidate": candidate_gen,
                "baseline": baseline_gen,
                **result.to_dict(),
            },
        }
    )
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Play against a checkpoint in the terminal (handy for Connect 4)."""
    manager = CheckpointManager(args.run)
    checkpoint = manager.load(args.generation)
    if checkpoint is None:
        print(f"no checkpoint found in {args.run}")
        return 1

    from az.core.mcts import BatchedMCTS, MCTSConfig

    device = torch.device("cpu")
    game = make_game(checkpoint.game_name, **checkpoint.game_kwargs)
    net = load_net_from_checkpoint(checkpoint, device)
    mcts = BatchedMCTS(
        game,
        torch_evaluator(net, device),
        MCTSConfig(num_simulations=args.simulations),
    )

    state = game.initial_state()
    human_first = not args.engine_first
    ply = 0

    while not game.is_terminal(state):
        print("\n" + game.render(state))
        human_turn = (ply % 2 == 0) == human_first

        if human_turn:
            text = input("your move (or 'quit'): ").strip()
            if text.lower() in {"quit", "exit"}:
                return 0
            action = game.string_to_action(state, text)
            if action is None:
                print("could not parse that move; try again")
                continue
        else:
            result = mcts.search([state], add_root_noise=False)[0]
            action = result.best_action
            print(
                f"engine plays {game.action_to_string(state, action)} "
                f"(eval {result.root_value:+.2f})"
            )

        state = game.next_state(state, action)
        ply += 1

    print("\n" + game.render(state))
    value = game.terminal_value(state)
    print("draw" if value == 0 else f"player to move loses ({value:+.0f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="az", description="AlphaZero from scratch")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run a training session")
    _add_common(train)
    train.add_argument("--game", default="chess", choices=["chess", "connect4"])
    train.add_argument(
        "--profile", default="balanced", choices=["tiny", "balanced", "strong"]
    )
    train.add_argument("--minutes", type=float, help="wall-clock budget")
    train.add_argument("--generations", type=int, help="generation budget")
    train.add_argument("--device", help="cuda / cpu (auto-detected by default)")
    train.add_argument("--simulations", type=int, help="override MCTS simulations")
    train.add_argument("--parallel", type=int, help="override parallel self-play games")
    train.add_argument(
        "--profile-override",
        action="store_true",
        help="allow --simulations/--parallel to change an existing run",
    )
    train.add_argument("--selfplay-fraction", type=float, default=0.8)
    train.set_defaults(func=cmd_train)

    status = sub.add_parser("status", help="show run progress")
    _add_common(status)
    status.add_argument("--last", type=int, default=15)
    status.set_defaults(func=cmd_status)

    evaluate = sub.add_parser("eval", help="play two checkpoints against each other")
    _add_common(evaluate)
    evaluate.add_argument("--candidate", type=int, help="defaults to newest")
    evaluate.add_argument("--baseline", type=int, help="defaults to candidate - gap")
    evaluate.add_argument("--gap", type=int, default=10)
    evaluate.add_argument("--games", type=int, default=40)
    evaluate.add_argument("--simulations", type=int, default=200)
    evaluate.add_argument("--parallel", type=int, default=16)
    evaluate.add_argument("--device")
    evaluate.add_argument("--seed", type=int)
    evaluate.set_defaults(func=cmd_eval)

    play = sub.add_parser("play", help="play against a checkpoint in the terminal")
    _add_common(play)
    play.add_argument("--generation", type=int)
    play.add_argument("--simulations", type=int, default=200)
    play.add_argument("--engine-first", action="store_true")
    play.set_defaults(func=cmd_play)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and not args.minutes and not args.generations:
        parser.error("train needs --minutes and/or --generations")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
