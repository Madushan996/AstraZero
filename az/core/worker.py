"""Run several self-play processes inside one container.

Why this exists: MCTS tree descent is single-threaded Python (python-chess move
generation dominates), so one self-play process saturates exactly one CPU core no
matter how many the container has. Measured evidence for that: throughput was almost
independent of batch width, and an RTX 4090 produced only ~16% more games per hour than
a T4 despite being far faster at inference. The GPU is not the bottleneck; one Python
thread is.

So a container with 4 cores and a GPU running a single self-play process wastes three
cores AND most of the GPU, while paying for all of it. Running one process per core
multiplies throughput for the same container cost, which is the single largest lever on
cost per game.

Each process writes its OWN shard, so there is no coordination and no lock contention --
the same property that lets many containers share one volume safely.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def run_selfplay_shard(
    run_dir: str,
    worker_tag: str,
    seconds: float,
    num_games: int,
    simulations: int = 0,
    parallel: int = 0,
    seed: Optional[int] = None,
    device_preference: str = "cuda",
    progress_every: int = 10,
    summary_path: Optional[str] = None,
) -> dict[str, Any]:
    """Play games and write one shard. Module-level so it survives spawn pickling."""
    import torch

    from az.core.checkpoint import CheckpointManager, load_net_from_checkpoint
    from az.core.game import make_game
    from az.core.mcts import torch_evaluator
    from az.core.pipeline import PipelineConfig
    from az.core.replay import ReplayBuffer
    from az.core.selfplay import play_games

    # Each process gets one core's worth of torch threads. Without this every process
    # spawns as many intra-op threads as there are cores, and they thrash each other.
    torch.set_num_threads(1)

    manager = CheckpointManager(Path(run_dir))
    config = PipelineConfig.from_dict(manager.read_config())

    from az.core.device import select_device

    device = select_device(device_preference)
    use_cuda = device.type == "cuda"

    checkpoint = manager.load(map_location="cpu")
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint in {run_dir}")

    game = make_game(checkpoint.game_name, **checkpoint.game_kwargs)

    def build(target: torch.device):
        model = load_net_from_checkpoint(checkpoint, target)
        # Force a real forward pass now. torch.cuda.is_available() can be True on a GPU
        # whose architecture this build has no compiled kernels for, and the failure
        # (cudaErrorNoKernelImageForDevice) only surfaces when a kernel actually runs --
        # by which point self-play is under way and the whole shard is lost.
        channels, height, width = game.observation_shape
        probe = torch.zeros((1, channels, height, width), device=target)
        mask = torch.ones((1, game.action_size), dtype=torch.bool, device=target)
        model.predict(probe, mask)
        return model

    try:
        net = build(device)
    except Exception as error:
        if device.type != "cuda":
            raise
        # Self-play is bound by Python tree search, not the GPU, so CPU is perhaps 20-30%
        # slower here -- vastly better than producing nothing.
        print(f"[worker] GPU unusable ({type(error).__name__}: {str(error)[:120]}); "
              f"falling back to CPU", flush=True)
        device = torch.device("cpu")
        use_cuda = False
        net = build(device)

    selfplay_config = config.selfplay_config()
    selfplay_config.num_games = num_games
    selfplay_config.max_seconds = seconds
    selfplay_config.seed = seed if seed is not None else (os.getpid() * 7919) % 1_000_003
    if simulations > 0:
        selfplay_config.num_simulations = simulations
    if parallel > 0:
        selfplay_config.parallel_games = parallel

    # Report progress as games land. A generation can run for half an hour, and without
    # this a long session shows nothing between generation lines -- indistinguishable
    # from a hang.
    started_at = time.time()

    def report(count: int, record) -> None:
        if count % progress_every == 0:
            rate = count / max(time.time() - started_at, 1e-6) * 3600
            print(
                f"  [{worker_tag}] {count} games "
                f"({(time.time() - started_at) / 60:.1f} min, {rate:.0f}/hr)",
                flush=True,
            )

    result = play_games(
        game,
        torch_evaluator(net, device, use_amp=use_cuda),
        selfplay_config,
        generation=checkpoint.run_state.generation,
        on_game_finished=report if progress_every > 0 else None,
    )

    if result.records:
        ReplayBuffer(manager.games_dir).add_games(
            result.records,
            generation=checkpoint.run_state.generation,
            worker=worker_tag,
        )

    summary = {"worker": worker_tag, **result.summary()}
    # Written to a file rather than returned on stdout so the parent can let the child's
    # progress stream straight through to the console.
    if summary_path:
        Path(summary_path).write_text(json.dumps(summary), encoding="utf-8")
    return summary


def run_parallel_selfplay(
    run_dir: str,
    worker_id: int,
    processes: int,
    seconds: float,
    num_games: int,
    simulations: int = 0,
    parallel: int = 0,
    device_preference: str = "cuda",
    progress_every: int = 10,
) -> dict[str, Any]:
    """Fan out `processes` self-play workers inside this container and aggregate.

    Uses plain subprocesses rather than `multiprocessing`. Serverless runtimes hold
    open gRPC file descriptors, and `mp` spawn tries to pass the parent's fd table to
    each child -- on Beam that fails outright with "bad value(s) in fds_to_keep".
    A fresh interpreter started via subprocess inherits nothing and works identically
    on Beam, Modal and a laptop.

    Each child writes its own shard, so results are collected from disk rather than
    passed back through IPC; the JSON on stdout is only for reporting.
    """
    if processes <= 1:
        summary = run_selfplay_shard(
            run_dir=run_dir,
            worker_tag=f"w{worker_id:03d}p0",
            seconds=seconds,
            num_games=num_games,
            simulations=simulations,
            parallel=parallel,
            device_preference=device_preference,
        )
        return {"processes": 1, "shards": [summary], **_aggregate([summary])}

    started = time.time()
    # Each child takes a slice of the game budget; the time budget is shared because
    # they run concurrently.
    per_process_games = max(1, num_games // processes)

    summaries_dir = Path(run_dir) / "_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    children = []
    summary_files = []
    for index in range(processes):
        tag = f"w{worker_id:03d}p{index}"
        summary_file = summaries_dir / f"{tag}.json"
        summary_file.unlink(missing_ok=True)
        summary_files.append(summary_file)

        command = [
            sys.executable,
            "-m",
            "az.core.worker",
            "--run-dir", run_dir,
            "--worker-tag", tag,
            "--seconds", str(seconds),
            "--num-games", str(per_process_games),
            "--simulations", str(simulations),
            "--parallel", str(parallel),
            "--seed", str((worker_id * 1000 + index) * 7919 + int(time.time()) % 9973),
            "--device", device_preference,
            "--summary-path", str(summary_file),
            "--progress-every", str(progress_every),
        ]
        environment = dict(os.environ)
        # Keep each child to one core's worth of torch threads, or they thrash.
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        children.append(
            subprocess.Popen(
                command,
                # Inherit stdout/stderr so progress appears live. A generation can run
                # for half an hour; buffering it until the end makes a working session
                # look identical to a hung one.
                stdout=None,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
        )

    summaries: list[dict[str, Any]] = []
    # Generous slack over the self-play budget for model load and the deliberate
    # overrun that lets in-flight games finish.
    budget = seconds * 3 + 600
    for child, summary_file in zip(children, summary_files):
        try:
            _, stderr = child.communicate(timeout=budget)
        except subprocess.TimeoutExpired:
            child.kill()
            _, stderr = child.communicate()
            summaries.append({"error": "timed out", "games": 0, "positions": 0})
            continue

        if summary_file.exists():
            try:
                summaries.append(json.loads(summary_file.read_text(encoding="utf-8")))
                continue
            except json.JSONDecodeError:
                pass

        summaries.append(
            {
                "error": _describe_failure(stderr, child.returncode),
                "games": 0,
                "positions": 0,
            }
        )

    return {
        "processes": processes,
        "wall_seconds": round(time.time() - started, 1),
        "shards": summaries,
        **_aggregate(summaries),
    }


SUMMARY_PREFIX = "__SELFPLAY_SUMMARY__ "


def _describe_failure(stderr: str, returncode: Optional[int]) -> str:
    """Build a diagnosable one-line error from a child's stderr.

    Truncating a traceback from either end loses the part that matters: the head is
    boilerplate frames and the tail may be cut mid-line. The final `SomeError: message`
    line is what actually identifies the fault, so pull that out explicitly. Learned the
    hard way -- a generation lost half its workers and the log had no exception in it.
    """
    lines = [line.rstrip() for line in (stderr or "").splitlines() if line.strip()]
    if not lines:
        if returncode is not None and returncode < 0:
            return f"child killed by signal {-returncode} (likely OOM)"
        return f"no output, exit code {returncode}"

    # The last line of a Python traceback is the exception itself.
    exception = lines[-1]
    for line in reversed(lines):
        if ("Error" in line or "Exception" in line) and ":" in line:
            exception = line
            break

    context = " | ".join(lines[-4:])
    return f"exit {returncode}: {exception[:200]} || {context[-400:]}"


def _parse_summary(stdout: str) -> Optional[dict[str, Any]]:
    """Pull the summary line out of a child's stdout, ignoring library chatter."""
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(SUMMARY_PREFIX):
            try:
                return json.loads(line[len(SUMMARY_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def _aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    games = sum(s.get("games", 0) for s in summaries)
    positions = sum(s.get("positions", 0) for s in summaries)
    failures = [s["error"] for s in summaries if s.get("error")]
    payload: dict[str, Any] = {"games": games, "positions": positions}
    if failures:
        payload["failures"] = failures
    return payload


def _main() -> int:
    """Child entry point: play one shard's worth of games and report on stdout."""
    import argparse

    parser = argparse.ArgumentParser(description="one self-play shard")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--worker-tag", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-path")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    summary = run_selfplay_shard(
        run_dir=args.run_dir,
        worker_tag=args.worker_tag,
        seconds=args.seconds,
        num_games=args.num_games,
        simulations=args.simulations,
        parallel=args.parallel,
        seed=args.seed,
        device_preference=args.device,
        progress_every=args.progress_every,
        summary_path=args.summary_path,
    )
    print(SUMMARY_PREFIX + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
