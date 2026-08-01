"""Run training on Beam (platform.beam.cloud).

Same pipeline as modal_app.py, different serverless platform. Everything in `az/` is
untouched -- only this orchestration layer differs, which is the point of keeping the
core platform-agnostic.

The shape is identical to the Modal version: self-play fans out across many cheap GPU
containers, training runs once per generation on one larger GPU, and all state lives on
a persistent Volume so sessions resume across days.

Authenticate first (in your own terminal, so your token stays yours):

    beam config create default --token YOUR_TOKEN

Quick start
-----------
    # one-time: copy existing games in, then create the run
    beam run beam_app.py:seed_volume
    beam run beam_app.py:init_run

    # a training session
    python beam_session.py --minutes 180 --workers 10 --simulations 100 --parallel 16

    # check progress
    beam run beam_app.py:run_status

Notes on differences from Modal
-------------------------------
* Beam Volumes are mounted filesystems with no explicit commit/reload step.
* `.map()` passes ONE argument per input, so workers take a single dict payload.
* There is no `beam volume cp` in this SDK, so the volume is seeded from the working
  directory that Beam syncs into the container.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from beam import Image, Volume, function

VOLUME_NAME = "alphazero-runs"
VOLUME_MOUNT = "./az_runs"

# Beam does not offer T4/L4 here -- only A10G and RTX 4090 are accepted. The RTX 4090
# is the faster of the two and usually the cheaper, so it is used for both roles.
#
# Worth knowing: self-play is bound by Python tree search, not by GPU throughput, so
# paying 4090 prices for it is less efficient than Modal's cheap T4 fan-out. Compare the
# measured cost per game on both platforms before committing a long run to either.
SELFPLAY_GPU = "RTX4090"
TRAIN_GPU = "RTX4090"

# Longer than Modal's 300s. Measured: a 15-minute Beam session budgeted 306s of
# self-play but took 640s wall, so roughly 3 minutes per generation goes to container
# startup, image load and checkpoint load before any game is played. Short generations
# spend most of their cost on that overhead, so amortise it over longer ones.
MIN_GENERATION_SECONDS = 600.0

# Seed directory synced from the working tree (see seed_volume below).
SEED_DIR = "beam_seed"

image = Image(
    python_version="python3.11",
    python_packages=["torch>=2.5", "numpy>=1.24", "python-chess>=1.10"],
)

volume = Volume(name=VOLUME_NAME, mount_path=VOLUME_MOUNT)


def _run_dir(run_name: str) -> Path:
    return Path(VOLUME_MOUNT) / run_name


# --------------------------------------------------------------------------- setup


@function(image=image, volumes=[volume], cpu=2, memory=4096, timeout=1800)
def seed_volume(run_name: str = "chess") -> dict[str, Any]:
    """Copy an existing run's games and config from the synced working directory.

    Only games and config are moved. Network weights are deliberately left behind --
    they are regenerable by training on the games, and the games are three orders of
    magnitude smaller. This is the same property that lets you change architecture
    mid-project without losing history.
    """
    source = Path(SEED_DIR)
    if not source.exists():
        return {"status": f"no {SEED_DIR}/ directory was synced; nothing to seed"}

    target = _run_dir(run_name)
    (target / "games").mkdir(parents=True, exist_ok=True)

    copied = 0
    for shard in (source / "games").glob("*.jsonl.gz"):
        destination = target / "games" / shard.name
        if not destination.exists():
            shutil.copy2(shard, destination)
            copied += 1

    for name in ("config.json", "history.jsonl"):
        candidate = source / name
        if candidate.exists() and not (target / name).exists():
            shutil.copy2(candidate, target / name)

    total = len(list((target / "games").glob("*.jsonl.gz")))
    return {
        "status": "seeded",
        "shards_copied": copied,
        "shards_total": total,
        "run_dir": str(target),
    }


@function(image=image, volumes=[volume], cpu=2, memory=8192, timeout=1800)
def init_run(
    run_name: str = "chess",
    game: str = "chess",
    profile: str = "balanced",
) -> dict[str, Any]:
    """Create the run directory and a generation-0 checkpoint with random weights."""
    from az.core.pipeline import TrainingSession, default_config

    run_dir = _run_dir(run_name)
    existing_config = run_dir / "config.json"

    if (run_dir / "latest.json").exists():
        session = TrainingSession(run_dir, device="cpu")
        return {
            "status": "already initialised",
            "generation": session.run_state.generation,
            "games": session.run_state.total_games,
        }

    if existing_config.exists():
        # Seeded from another platform: keep that run's definition so the games in the
        # buffer stay compatible with the encoding they were generated under.
        config = None
        print("using seeded config.json")
    else:
        config = default_config(game, profile=profile)

    session = TrainingSession(run_dir, config=config, device="cpu")
    session.manager.save(
        session.net,
        session.run_state,
        session.config.game_name,
        session.config.game_kwargs,
        optimizer=session.optimizer,
    )

    return {
        "status": "created",
        "run": run_name,
        "game": session.config.game_name,
        "parameters": session.net.parameter_count(),
        "seeded_shards": len(list((run_dir / "games").glob("*.jsonl.gz"))),
    }


# ------------------------------------------------------------------------ self-play


@function(
    image=image,
    volumes=[volume],
    gpu=SELFPLAY_GPU,
    # MCTS tree descent is pure Python, so a starved worker leaves its GPU idle.
    cpu=4,
    memory=4096,
    timeout=3600,
    retries=1,
)
def selfplay_worker(payload: dict) -> dict[str, Any]:
    """Generate games with the current checkpoint, one shard per inner process.

    Beam's `.map()` passes exactly one argument per input, so everything arrives in a
    single dict rather than as positional arguments.

    `processes` fans out inside the container. MCTS tree descent is single-threaded
    Python, so one process leaves the other cores -- and most of the GPU -- idle while
    you pay for them. See az/core/worker.py.
    """
    from az.core.worker import run_parallel_selfplay

    run_name = payload.get("run_name", "chess")
    worker_id = int(payload.get("worker_id", 0))

    return {
        "worker": worker_id,
        **run_parallel_selfplay(
            run_dir=str(_run_dir(run_name)),
            worker_id=worker_id,
            processes=int(payload.get("processes", 4)),
            seconds=float(payload.get("seconds", 600.0)),
            num_games=int(payload.get("num_games", 200)),
            simulations=int(payload.get("simulations", 0)),
            parallel=int(payload.get("parallel", 0)),
        ),
    }


# -------------------------------------------------------------------------- training


@function(
    image=image,
    volumes=[volume],
    gpu=TRAIN_GPU,
    cpu=4,
    memory=16384,
    timeout=3600,
)
def train_generation(
    run_name: str = "chess",
    new_games: int = 0,
    new_positions: int = 0,
    selfplay_seconds: float = 0.0,
    steps: int = 0,
) -> dict[str, Any]:
    """Train on the current replay window and write the next checkpoint."""
    from az.core.pipeline import TrainingSession
    from az.core.replay import materialize
    from az.core.trainer import train_on_data

    # Never hardcode "cuda": a GPU-declared container can still land without one when
    # the account's GPU quota is saturated, and crashing there loses the generation's
    # games. Training on CPU is slow but it is not a failure.
    session = TrainingSession(_run_dir(run_name), device=None)
    train_config = session.config.train_config()
    generation = session.run_state.generation

    # Training is ~2% of a session's cost while self-play is ~97%, so under-training is
    # a far more expensive mistake than over-training. Sanity check: steps x batch_size
    # should be a few times the positions each generation adds, or the network never
    # catches up with its own game production and the loss plateaus.
    if steps > 0:
        train_config.steps_per_generation = steps

    records = session.buffer.load_window(max_plies=train_config.max_positions * 2)
    if not records:
        return {"status": "no games yet", "generation": generation}

    data = materialize(
        session.game, records, max_positions=train_config.max_positions
    )
    metrics = train_on_data(
        session.net,
        session.optimizer,
        data,
        train_config,
        session.device,
        generation=generation,
        global_step=session.run_state.global_step,
        scaler=session.scaler,
    )

    session.run_state.global_step += metrics.steps
    session.run_state.total_train_seconds += metrics.seconds
    session.run_state.generation += 1
    session.run_state.total_games += new_games
    session.run_state.total_positions += new_positions
    session.run_state.total_selfplay_seconds += selfplay_seconds

    session.manager.save(
        session.net,
        session.run_state,
        session.config.game_name,
        session.config.game_kwargs,
        optimizer=session.optimizer,
        scaler=session.scaler,
    )
    entry = {
        "generation": generation,
        "train": metrics.to_dict(),
        "window_games": len(records),
        "buffer": session.buffer.stats(),
        "train_data_mb": round(data.nbytes() / 1e6, 1),
    }
    session.manager.append_history(entry)
    session.housekeeping()
    return entry


# ------------------------------------------------------------------------ evaluation


@function(
    image=image, volumes=[volume], gpu=TRAIN_GPU, cpu=4, memory=8192, timeout=3600
)
def evaluate_pair(
    run_name: str = "chess",
    candidate: Optional[int] = None,
    baseline: Optional[int] = None,
    gap: int = 10,
    games: int = 60,
    simulations: int = 200,
    random_opening_plies: int = 10,
) -> dict[str, Any]:
    """Play two checkpoints head to head -- the only honest progress signal."""
    import torch

    from az.core.arena import play_match
    from az.core.checkpoint import CheckpointManager, load_net_from_checkpoint
    from az.core.game import make_game
    from az.core.mcts import torch_evaluator

    manager = CheckpointManager(_run_dir(run_name))
    generations = manager.available_generations()
    if len(generations) < 2:
        return {"status": f"need 2 checkpoints, have {len(generations)}"}

    candidate = candidate if candidate is not None else generations[-1]
    if baseline is None:
        older = [g for g in generations if g <= candidate - gap]
        baseline = older[-1] if older else generations[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_ckpt = manager.load(candidate, map_location=str(device))
    baseline_ckpt = manager.load(baseline, map_location=str(device))

    # Housekeeping deletes old checkpoints, so a perfectly reasonable-looking baseline
    # may simply no longer exist. Say so, with the list of what does.
    missing = [
        generation
        for generation, checkpoint in ((candidate, candidate_ckpt), (baseline, baseline_ckpt))
        if checkpoint is None
    ]
    if missing:
        return {
            "status": f"checkpoint(s) not found: {missing}",
            "available": generations,
            "hint": "old generations are pruned; pick a baseline from 'available'",
        }

    game = make_game(candidate_ckpt.game_name, **candidate_ckpt.game_kwargs)
    result = play_match(
        game,
        torch_evaluator(load_net_from_checkpoint(candidate_ckpt, device), device),
        torch_evaluator(load_net_from_checkpoint(baseline_ckpt, device), device),
        num_games=games,
        num_simulations=simulations,
        parallel_games=32,
        random_opening_plies=random_opening_plies,
    )

    payload = {"candidate": candidate, "baseline": baseline, **result.to_dict()}
    manager.append_history({"generation": candidate, "arena": payload})
    return payload


@function(image=image, volumes=[volume], cpu=1, memory=2048, timeout=600)
def run_status(run_name: str = "chess", last: int = 10) -> dict[str, Any]:
    from az.core.checkpoint import CheckpointManager

    manager = CheckpointManager(_run_dir(run_name))
    history = manager.read_history()
    payload = {
        "summary": manager.summary(),
        "recent": history[-last:],
        "evaluations": [e["arena"] for e in history if "arena" in e][-10:],
    }
    print(json.dumps(payload, indent=2, default=str))
    return payload


@function(image=image, volumes=[volume], cpu=1, memory=2048, timeout=21600)
def run_session_remote(
    run_name: str = "chess",
    minutes: float = 120.0,
    # Beam caps concurrent GPU containers per account. Six RTX 4090 workers hit
    # "gpu quota exceeded" on this plan; four is known good. Raise only after checking
    # your quota, and remember train_generation needs a GPU slot too.
    workers: int = 4,
    games_per_worker: int = 400,
    selfplay_fraction: float = 0.85,
    simulations: int = 0,
    parallel: int = 0,
    processes: int = 4,
) -> dict[str, Any]:
    """Run the whole generation loop inside Beam.

    KNOWN LIMITATION -- prefer beam_session.py for real runs.

    Containers launched from inside another container do not reliably receive the
    client's file sync. Observed: a 4-worker generation where 2 workers died with
    `ModuleNotFoundError: No module named 'beam_app'` while the other 2 (warm containers
    left over from an earlier client-driven run) worked fine. Throughput silently halved.

    The client-side driver in beam_session.py syncs on every dispatch, so every worker
    gets the code. The cost is that the laptop must stay awake for the session. Modal
    does not have this problem, which is why `detached` mode works there.
    """
    deadline = time.time() + minutes * 60
    generations: list[dict[str, Any]] = []
    consecutive_failures = 0

    print(f"remote session: {minutes:.0f} min, {workers} workers x {processes} procs")

    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < MIN_GENERATION_SECONDS:
            print(f"{remaining:.0f}s left -- too short for a useful generation")
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
            # GPU quota is shared and can be briefly unavailable. Losing a whole
            # multi-hour session to a transient scheduling error would be far worse
            # than skipping one generation, so back off and carry on.
            print(f"  self-play dispatch failed: {type(error).__name__}: {error}")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("  three dispatch failures in a row; stopping")
                break
            time.sleep(30)
            continue

        games = sum(r.get("games", 0) for r in results)
        positions = sum(r.get("positions", 0) for r in results)
        elapsed = time.time() - started

        failures = [f for r in results for f in r.get("failures", [])]
        print(
            f"  self-play: {games} games / {positions} positions in {elapsed:.0f}s"
            + (f" ({len(failures)} child failures)" if failures else "")
        )
        if games == 0:
            consecutive_failures += 1
            print(f"  no games produced. failures: {failures[:2]}")
            if consecutive_failures >= 2:
                print("  giving up rather than burning the budget on empty generations")
                break
            continue

        consecutive_failures = 0

        try:
            entry = train_generation.remote(
                run_name=run_name,
                new_games=games,
                new_positions=positions,
                selfplay_seconds=elapsed,
            )
        except Exception as error:
            # The games are already durable on the volume, so a failed training step
            # costs one generation, not the data. Keep going; the next generation
            # trains on these games too.
            print(f"  training failed: {type(error).__name__}: {error}")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue

        train = (entry or {}).get("train", {})
        holdout = train.get("holdout") or {}
        print(
            f"  trained: loss {train.get('loss', 0):.3f} "
            f"(p {train.get('policy_loss', 0):.3f} v {train.get('value_loss', 0):.3f})"
            + (
                f" | gap p {holdout.get('policy_gap', 0):+.3f}"
                if holdout
                else ""
            )
        )
        generations.append({"games": games, "positions": positions, **train})

    total = sum(g["games"] for g in generations)
    print(f"session done: {len(generations)} generations, {total} games")
    return {
        "generations": len(generations),
        "games": total,
        "positions": sum(g["positions"] for g in generations),
        "detail": generations,
    }


@function(image=image, volumes=[volume], cpu=0.5, memory=1024, timeout=180)
def probe_nested_call(run_name: str = "chess") -> dict[str, Any]:
    """Can a container invoke another Beam function?

    That single fact decides whether a server-side session driver is possible here.
    Modal allows it, which is how `detached` mode works there. If Beam does not, the
    session loop has to run on the client and the laptop must stay awake.
    """
    try:
        result = run_status.remote(run_name=run_name, last=1)
        return {"nested_calls": "work", "returned": type(result).__name__}
    except Exception as error:
        return {"nested_calls": "fail", "error": f"{type(error).__name__}: {error}"[:400]}


@function(image=image, volumes=[volume], cpu=2, memory=8192, timeout=900)
def prepare_export(
    run_name: str = "chess", generation: Optional[int] = None
) -> dict[str, Any]:
    """Write a play-only checkpoint to the volume, ready to be fetched in chunks.

    Beam has no volume download in this SDK, so getting a checkpoint onto a laptop
    means streaming it back through function calls. Optimizer state is stripped first:
    it is two thirds of the file and is useless for playing, taking 145 MB down to ~48.
    """
    import hashlib

    import torch

    from az.core.checkpoint import CheckpointManager

    manager = CheckpointManager(_run_dir(run_name))
    generation = generation if generation is not None else manager.latest_generation()
    if generation is None:
        return {"status": "no checkpoints yet"}

    checkpoint = manager.load(generation, map_location="cpu")
    if checkpoint is None:
        return {
            "status": f"generation {generation} not found",
            "available": manager.available_generations(),
        }

    export_dir = _run_dir(run_name) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"gen{generation:05d}_play.pt"

    torch.save(
        {
            "net_config": checkpoint.net_config.to_dict(),
            "game_name": checkpoint.game_name,
            "game_kwargs": checkpoint.game_kwargs,
            "run_state": checkpoint.run_state.to_dict(),
            "model_state": checkpoint.model_state,
            "optimizer_state": None,
            "scaler_state": None,
            "rng_state": {},
            "extra": {"play_only": True},
        },
        path,
    )

    data = path.read_bytes()
    return {
        "status": "ready",
        "generation": generation,
        "path": str(path),
        "bytes": len(data),
        "mb": round(len(data) / 1e6, 1),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


@function(image=image, volumes=[volume], cpu=2, memory=8192, timeout=1800)
def prepare_games_bundle(run_name: str = "chess") -> dict[str, Any]:
    """Tar the replay buffer for transfer to another platform.

    The games are the part worth moving: network weights regenerate by training on
    them, but a game once discarded is gone. Shards are already gzipped, so the tar is
    stored uncompressed -- recompressing would cost CPU for nothing.
    """
    import hashlib
    import tarfile

    run_dir = _run_dir(run_name)
    games_dir = run_dir / "games"
    if not games_dir.exists():
        return {"status": "no games directory"}

    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / "games.tar"

    shards = sorted(games_dir.glob("*.jsonl.gz"))
    with tarfile.open(path, "w") as archive:
        for shard in shards:
            archive.add(shard, arcname=f"games/{shard.name}")

    data = path.read_bytes()
    config_path = run_dir / "config.json"
    return {
        "status": "ready",
        "name": "games.tar",
        "shards": len(shards),
        "bytes": len(data),
        "mb": round(len(data) / 1e6, 1),
        "sha256": hashlib.sha256(data).hexdigest(),
        "config": config_path.read_text(encoding="utf-8") if config_path.exists() else None,
    }


@function(image=image, volumes=[volume], cpu=1, memory=4096, timeout=600)
def fetch_file_chunk(
    run_name: str = "chess",
    name: str = "",
    offset: int = 0,
    length: int = 4_000_000,
) -> dict[str, Any]:
    """Return one base64-encoded byte range of a file in the run's export directory."""
    import base64

    path = _run_dir(run_name) / "export" / name
    if not path.exists():
        return {"status": f"{name} not prepared"}

    with path.open("rb") as handle:
        handle.seek(offset)
        block = handle.read(length)

    return {
        "status": "ok",
        "offset": offset,
        "length": len(block),
        "data": base64.b64encode(block).decode("ascii"),
    }


@function(image=image, volumes=[volume], cpu=1, memory=4096, timeout=600)
def fetch_export_chunk(
    run_name: str = "chess",
    generation: int = 0,
    offset: int = 0,
    length: int = 4_000_000,
) -> dict[str, Any]:
    """Return one base64-encoded byte range of the prepared checkpoint export."""
    return fetch_file_chunk.local(
        run_name=run_name,
        name=f"gen{generation:05d}_play.pt",
        offset=offset,
        length=length,
    )


@function(image=image, volumes=[volume], cpu=1, memory=2048, timeout=600)
def export_checkpoint(run_name: str = "chess", generation: Optional[int] = None) -> dict:
    """Report where the newest checkpoint lives, for pulling down to play in a GUI."""
    from az.core.checkpoint import CheckpointManager

    manager = CheckpointManager(_run_dir(run_name))
    generation = generation if generation is not None else manager.latest_generation()
    if generation is None:
        return {"status": "no checkpoints yet"}

    path = manager.generation_dir(generation) / "model.pt"
    return {
        "generation": generation,
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else 0,
    }
