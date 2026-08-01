"""Drive a Beam training session from your machine.

    python beam_session.py --minutes 180 --workers 10 --simulations 100 --parallel 16

Unlike the Modal version there is no server-side driver here: Beam functions are invoked
from the client, so this loop runs locally and your machine needs to stay awake for the
session. Every completed generation is already checkpointed to the Volume, so an
interruption costs at most the current generation's in-flight games -- rerun the same
command and it resumes.
"""

from __future__ import annotations

import argparse
import time

from beam_app import MIN_GENERATION_SECONDS, selfplay_worker, train_generation


def run_session(
    run_name: str,
    minutes: float,
    workers: int,
    games_per_worker: int,
    simulations: int,
    parallel: int,
    selfplay_fraction: float,
    processes: int,
    train_steps: int,
) -> None:
    deadline = time.time() + minutes * 60
    generation_index = 0
    failures = 0

    print(
        f"session: {minutes:.0f} min, {workers} workers x {processes} processes, "
        f"run={run_name}"
    )

    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < MIN_GENERATION_SECONDS:
            print(f"  {remaining:.0f}s left -- too short for a useful generation")
            break

        generation_seconds = min(
            remaining, max(MIN_GENERATION_SECONDS, remaining / 3 + 60)
        )
        selfplay_seconds = generation_seconds * selfplay_fraction

        payloads = [
            {
                "run_name": run_name,
                "worker_id": worker_id,
                "seconds": selfplay_seconds,
                "num_games": games_per_worker,
                "simulations": simulations,
                "parallel": parallel,
                "processes": processes,
            }
            for worker_id in range(workers)
        ]

        started = time.time()
        try:
            results = [r for r in selfplay_worker.map(payloads) if isinstance(r, dict)]
        except Exception as error:
            # GPU quota is shared and can be briefly unavailable. Losing a multi-hour
            # session to one scheduling hiccup is far worse than skipping a generation.
            print(f"  dispatch failed: {type(error).__name__}: {error}")
            failures += 1
            if failures >= 3:
                print("  three dispatch failures in a row; stopping")
                break
            time.sleep(30)
            continue

        games = sum(r.get("games", 0) for r in results)
        positions = sum(r.get("positions", 0) for r in results)
        elapsed = time.time() - started

        child_failures = [f for r in results for f in r.get("failures", [])]
        print(
            f"  gen {generation_index}: {games} games / {positions} positions "
            f"from {len(results)}/{workers} workers in {elapsed:.0f}s"
            + (f" ({len(child_failures)} child failures)" if child_failures else "")
        )
        if child_failures:
            for failure in child_failures[:2]:
                print(f"    ! {failure}")

        if games == 0:
            failures += 1
            print(
                "  no games produced. If there are no child failures above, lower "
                "--parallel or --simulations so a batch fits the time budget."
            )
            if failures >= 2:
                break
            continue

        failures = 0

        try:
            entry = train_generation.remote(
                run_name=run_name,
                new_games=games,
                new_positions=positions,
                selfplay_seconds=elapsed,
                steps=train_steps,
            )
        except Exception as error:
            # The games are already durable on the volume, so this costs one training
            # step, not the data. The next generation trains on these games too.
            print(f"  training failed: {type(error).__name__}: {error}")
            failures += 1
            if failures >= 3:
                break
            continue
        train = (entry or {}).get("train", {})
        holdout = train.get("holdout")
        line = (
            f"  gen {generation_index}: loss {train.get('loss', 0):.3f} "
            f"(p {train.get('policy_loss', 0):.3f} v {train.get('value_loss', 0):.3f})"
        )
        if holdout:
            line += (
                f" | gap p {holdout.get('policy_gap', 0):+.3f} "
                f"v {holdout.get('value_gap', 0):+.3f}"
            )
        print(line)
        generation_index += 1

    print(f"\nsession complete after {generation_index} generations.")
    print("check progress:  beam run beam_app.py:run_status")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Beam training session")
    parser.add_argument("--run-name", default="chess")
    parser.add_argument("--minutes", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--games-per-worker", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=0, help="0 = use config")
    parser.add_argument("--parallel", type=int, default=0, help="0 = use config")
    parser.add_argument("--selfplay-fraction", type=float, default=0.85)
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="self-play processes per container; match the container's cpu count",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help=(
            "override steps per generation (0 = use config). Aim for steps x batch_size "
            "to be a few times the positions each generation adds, or the loss plateaus"
        ),
    )
    args = parser.parse_args()

    run_session(
        run_name=args.run_name,
        minutes=args.minutes,
        workers=args.workers,
        games_per_worker=args.games_per_worker,
        simulations=args.simulations,
        parallel=args.parallel,
        selfplay_fraction=args.selfplay_fraction,
        processes=args.processes,
        train_steps=args.train_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
