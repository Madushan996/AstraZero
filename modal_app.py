"""Run training on Modal: fan-out self-play, single-GPU training, persistent Volume.

The shape of this file follows from one fact: self-play is the bottleneck, not the
gradient steps. So self-play is spread across many cheap containers running in parallel,
while training happens once per generation on a single larger GPU.

    generation N:
        [worker 0] [worker 1] ... [worker K]     <- parallel, cheap GPUs
              \\        |         /
               ---> shards on the Volume <---
                          |
                    [trainer, one GPU]           <- reads window, writes checkpoint

Because every worker writes its OWN shard file, concurrent writes to the Volume can
never conflict.

State lives entirely on a modal.Volume, so a session is genuinely resumable: run for
three hours today, close the laptop, run again next week, and it continues from the same
generation with the same replay buffer.

Quick start
-----------
    pip install modal
    modal setup

    # one-time: create the run
    modal run modal_app.py::init --game chess --profile balanced

    # a training session (repeat this whenever you like)
    modal run modal_app.py::session --minutes 180 --workers 20

    # see where things stand
    modal run modal_app.py::status

    # measure real improvement
    modal run modal_app.py::evaluate --gap 10 --games 60

    # pull a checkpoint down to play in a GUI
    modal volume get alphazero-runs chess/checkpoints/gen00042 ./runs/chess/checkpoints/
    modal volume get alphazero-runs chess/config.json ./runs/chess/
"""

from __future__ import annotations

import time
from typing import Any, Optional

import modal

APP_NAME = "alphazero-chess"
VOLUME_NAME = "alphazero-runs"
VOLUME_PATH = "/runs"

# Self-play is inference-bound and runs wide, so use the cheapest GPU that fits.
# Training is a single job on a larger one. Both are overridable per call.
SELFPLAY_GPU = "T4"
TRAIN_GPU = "L4"

# Shortest generation worth paying for, given ~40s of container startup per worker.
MIN_GENERATION_SECONDS = 300.0

# Longest single generation. Not a performance setting -- a blast radius. A worker only
# writes its shard when self-play RETURNS, so anything that kills a worker mid-generation
# destroys everything it played. Capping generation length bounds that loss and
# checkpoints progress more often. A 3-hour session used to front-load a single 52-minute
# generation; losing it cost the better part of an hour of ten workers.
MAX_GENERATION_SECONDS = 1800.0

# Wall-clock limit on one self-play worker. MUST comfortably exceed the longest possible
# self-play call: the soft deadline, times the overrun factor that lets in-flight games
# finish, plus container start and model load. Getting this wrong is silent and total --
# Modal kills the container and every game it played is lost, then `retries` runs it
# again to be killed again.
SELFPLAY_WORKER_TIMEOUT = 14400  # 4 hours
SELFPLAY_OVERRUN_FACTOR = 1.5
SELFPLAY_STARTUP_MARGIN = 300.0


def max_safe_selfplay_seconds() -> float:
    """Longest self-play budget a worker can be given without risking its own timeout."""
    return (SELFPLAY_WORKER_TIMEOUT - SELFPLAY_STARTUP_MARGIN) / SELFPLAY_OVERRUN_FACTOR

# Measured seconds of one worker process per (ply x simulation x concurrent game).
# Self-play is bound by single-threaded Python tree search, so this holds across GPUs.
SECONDS_PER_PLY_SIM_GAME = 0.0012
TYPICAL_PLIES = 180


def minimum_generation_seconds(
    simulations: int, parallel: int, selfplay_fraction: float = 0.85
) -> float:
    """How long a generation must run for a batch of games to actually finish.

    A worker plays `parallel` games at once and they complete together, so a generation
    shorter than one batch produces ZERO games while costing full price. This has bitten
    twice: `--parallel 96` at 100 sims needs 25 minutes per batch, and a 20-minute probe
    forced `--parallel 4` and looked like Modal was slow when it was the pacing.

    Returns the generation length needed, with 30% headroom.
    """
    batch_seconds = SECONDS_PER_PLY_SIM_GAME * TYPICAL_PLIES * simulations * parallel
    return max(MIN_GENERATION_SECONDS, batch_seconds * 1.3 / max(selfplay_fraction, 0.1))

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pin these to exact versions once your first session works, so a surprise upgrade
    # can never change results midway through a months-long run.
    .pip_install("torch>=2.5", "numpy>=1.24", "python-chess>=1.10")
    .add_local_python_source("az")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run_dir(run_name: str):
    from pathlib import Path

    return Path(VOLUME_PATH) / run_name


# --------------------------------------------------------------------------- setup


@app.function(volumes={VOLUME_PATH: volume}, timeout=600)
def init_run(
    run_name: str = "chess",
    game: str = "chess",
    profile: str = "balanced",
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create the run directory and a generation-0 checkpoint with random weights.

    Workers always load a checkpoint rather than constructing a network themselves, so
    there is exactly one place the architecture is decided.
    """
    from az.core.pipeline import TrainingSession, default_config

    run_dir = _run_dir(run_name)
    if (run_dir / "config.json").exists():
        session = TrainingSession(run_dir, device="cpu")
        return {
            "status": "already exists",
            "generation": session.run_state.generation,
            "games": session.run_state.total_games,
        }

    config = default_config(game, profile=profile)
    for section, values in (overrides or {}).items():
        current = getattr(config, section, None)
        if isinstance(current, dict):
            current.update(values)
        else:
            setattr(config, section, values)

    session = TrainingSession(run_dir, config=config, device="cpu")
    # Persist generation 0 so self-play workers have weights to load immediately.
    session.manager.save(
        session.net,
        session.run_state,
        config.game_name,
        config.game_kwargs,
        optimizer=session.optimizer,
    )
    volume.commit()

    return {
        "status": "created",
        "run": run_name,
        "game": config.game_name,
        "profile": profile,
        "parameters": session.net.parameter_count(),
    }


# ------------------------------------------------------------------------ self-play


@app.function(
    gpu=SELFPLAY_GPU,
    # MCTS tree descent is pure Python (python-chess move generation), so a self-play
    # worker is CPU-bound as much as GPU-bound. Starving it of cores wastes the GPU it
    # is paying for. CPU is ~$0.05/core/hour, far cheaper than the idle GPU time it
    # buys back.
    cpu=4.0,
    memory=4096,
    volumes={VOLUME_PATH: volume},
    timeout=SELFPLAY_WORKER_TIMEOUT,
    retries=modal.Retries(max_retries=1),
)
def selfplay_worker(
    run_name: str,
    worker_id: int,
    seconds: float,
    num_games: int,
    simulations: int = 0,
    parallel: int = 0,
    processes: int = 4,
) -> dict[str, Any]:
    """Generate games with the current checkpoint, one shard per inner process.

    `simulations` and `parallel` override the stored config for this call only (0 means
    "use the config"). Search width is a speed/strength dial you will want to turn
    between sessions without touching the run definition -- unlike network sizing, it
    can change freely mid-run.

    `processes` fans out inside the container: MCTS tree descent is single-threaded
    Python, so one process per container leaves the other cores and most of the GPU
    idle while you pay for them. See az/core/worker.py.
    """
    from az.core.worker import run_parallel_selfplay

    volume.reload()  # pick up the checkpoint the trainer just wrote

    result = run_parallel_selfplay(
        run_dir=str(_run_dir(run_name)),
        worker_id=worker_id,
        processes=processes,
        seconds=seconds,
        num_games=num_games,
        simulations=simulations,
        parallel=parallel,
    )

    if result.get("games"):
        volume.commit()

    return {"worker": worker_id, **result}


# -------------------------------------------------------------------------- training


@app.function(
    gpu=TRAIN_GPU,
    # Replaying the window into observations is single-threaded Python; the uint8
    # observation array is ~2 GB at 250k chess positions, so give it real headroom.
    cpu=4.0,
    memory=16384,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def train_generation(
    run_name: str = "chess",
    new_games: int = 0,
    new_positions: int = 0,
    selfplay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Train on the current replay window and write the next checkpoint.

    The self-play totals are passed in by the driver because the games were produced by
    other containers -- counting the replay window here instead would re-count the same
    games every generation.
    """
    from az.core.pipeline import TrainingSession
    from az.core.replay import materialize
    from az.core.trainer import train_on_data

    volume.reload()

    session = TrainingSession(_run_dir(run_name), device="cuda")
    train_config = session.config.train_config()
    generation = session.run_state.generation

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
    volume.commit()
    return entry


# ------------------------------------------------------------------------ evaluation


@app.function(gpu=TRAIN_GPU, volumes={VOLUME_PATH: volume}, timeout=3600)
def evaluate_pair(
    run_name: str = "chess",
    candidate: Optional[int] = None,
    baseline: Optional[int] = None,
    gap: int = 10,
    games: int = 60,
    simulations: int = 200,
    random_opening_plies: int = 10,
) -> dict[str, Any]:
    """Play two checkpoints head to head. This is the only honest progress signal.

    `random_opening_plies` matters more than it looks. Two checkpoints a few generations
    apart are nearly the same network, and two near-identical networks playing greedily
    produce the same game every time -- almost always a draw, which measures nothing.
    Randomising more opening moves forces varied positions and yields decisive games,
    which is what actually carries Elo information.
    """
    import torch

    from az.core.arena import play_match
    from az.core.checkpoint import CheckpointManager, load_net_from_checkpoint
    from az.core.game import make_game
    from az.core.mcts import torch_evaluator

    volume.reload()

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

    # Housekeeping deletes old checkpoints, so a reasonable-looking baseline may simply
    # no longer exist. Say so, with the list of what does.
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

    payload = {
        "candidate": candidate,
        "baseline": baseline,
        "random_opening_plies": random_opening_plies,
        **result.to_dict(),
    }
    manager.append_history({"generation": candidate, "arena": payload})
    volume.commit()
    return payload


# --------------------------------------------------------------- server-side driver


@app.function(
    # Pure orchestration: no GPU, minimal CPU. The cost of this container over a long
    # session is a few cents.
    cpu=0.5,
    memory=1024,
    volumes={VOLUME_PATH: volume},
    timeout=28800,
)
def run_session_remote(
    run_name: str = "chess",
    minutes: float = 180.0,
    workers: int = 10,
    games_per_worker: int = 200,
    selfplay_fraction: float = 0.85,
    simulations: int = 0,
    parallel: int = 0,
    processes: int = 4,
) -> dict[str, Any]:
    """Run a whole session inside Modal, so the laptop is free to disconnect.

    Identical logic to the `session` local entrypoint, but the loop that decides when
    each generation starts lives on Modal rather than on your machine. Launch it with
    `.spawn()` and you can close the lid -- the session keeps going.
    """
    deadline = time.time() + minutes * 60
    generations: list[dict[str, Any]] = []

    floor = minimum_generation_seconds(
        simulations or 200, parallel or 16, selfplay_fraction
    )
    print(f"remote session: {minutes:.0f} min, {workers} workers, run={run_name}")
    print(f"minimum generation length for these settings: {floor:.0f}s")

    while time.time() < deadline:
        remaining = deadline - time.time()
        # Below the floor a batch of `parallel` games cannot finish, so the generation
        # would cost full price and produce nothing.
        if remaining < floor:
            print(f"{remaining:.0f}s left -- below the {floor:.0f}s floor, stopping")
            break

        generation_seconds = min(
            remaining, MAX_GENERATION_SECONDS, max(floor, remaining / 3 + 60)
        )
        # Hard bound: a worker given more than this can outlive its own timeout, and a
        # killed worker loses every game it played.
        selfplay_seconds = min(
            generation_seconds * selfplay_fraction, max_safe_selfplay_seconds()
        )

        started = time.time()
        results = list(
            selfplay_worker.starmap(
                [
                    (
                        run_name,
                        worker_id,
                        selfplay_seconds,
                        games_per_worker,
                        simulations,
                        parallel,
                        processes,
                    )
                    for worker_id in range(workers)
                ]
            )
        )
        games = sum(r.get("games", 0) for r in results)
        positions = sum(r.get("positions", 0) for r in results)
        elapsed = time.time() - started
        print(f"  self-play: {games} games / {positions} positions in {elapsed:.0f}s")

        entry = train_generation.remote(
            run_name=run_name,
            new_games=games,
            new_positions=positions,
            selfplay_seconds=elapsed,
        )
        train = entry.get("train", {})
        print(
            f"  trained: loss {train.get('loss', 0):.3f} "
            f"(p {train.get('policy_loss', 0):.3f} v {train.get('value_loss', 0):.3f})"
        )
        generations.append({"games": games, "positions": positions, **train})

    total_games = sum(g["games"] for g in generations)
    print(f"session done: {len(generations)} generations, {total_games} games")
    return {
        "generations": len(generations),
        "games": total_games,
        "positions": sum(g["positions"] for g in generations),
        "detail": generations,
    }


@app.function(cpu=4, memory=8192, volumes={VOLUME_PATH: volume}, timeout=3600)
def enable_adjudication(
    run_name: str = "chess", threshold: int = 5, relabel: bool = True
) -> dict[str, Any]:
    """Turn on material adjudication and retroactively fix existing game labels.

    Measured on this run: 32% of all games ended as draws with a piece or more on the
    board -- the engine wins material, cannot convert, shuffles to the 50-move rule, and
    the position is labelled 0.0. That teaches the value head that material is worthless.

    Because the buffer stores move lists rather than tensors, the outcome can simply be
    re-derived under the new rule. Every existing game is corrected in one pass instead
    of waiting for tens of thousands of mislabelled games to age out of the window.
    """
    import json

    from az.core.game import make_game
    from az.core.pipeline import PipelineConfig
    from az.core.relabel import relabel_directory

    volume.reload()

    run_dir = _run_dir(run_name)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {"status": f"no config.json in {run_dir}"}

    stored = json.loads(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(stored)
    previous = config.game_kwargs.get("adjudicate_material_at", 0)
    config.game_kwargs["adjudicate_material_at"] = threshold

    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8"
    )

    report: dict[str, Any] = {
        "status": "enabled",
        "threshold": threshold,
        "previous": previous,
    }

    if relabel:
        game = make_game(config.game_name, **config.game_kwargs)
        report["relabel"] = relabel_directory(game, run_dir / "games")

    volume.commit()
    return report


@app.function(volumes={VOLUME_PATH: volume}, timeout=300)
def run_status(run_name: str = "chess", last: int = 15) -> dict[str, Any]:
    from az.core.checkpoint import CheckpointManager

    volume.reload()
    manager = CheckpointManager(_run_dir(run_name))
    history = manager.read_history()
    return {
        "summary": manager.summary(),
        "recent": history[-last:],
        "evaluations": [e["arena"] for e in history if "arena" in e][-10:],
    }


# ----------------------------------------------------------------------- entrypoints


@app.local_entrypoint()
def init(run_name: str = "chess", game: str = "chess", profile: str = "balanced"):
    """Create a new run. Safe to call twice -- it will not overwrite."""
    print(init_run.remote(run_name=run_name, game=game, profile=profile))


@app.local_entrypoint()
def session(
    run_name: str = "chess",
    minutes: float = 180.0,
    workers: int = 20,
    games_per_worker: int = 200,
    selfplay_fraction: float = 0.85,
    simulations: int = 0,
    parallel: int = 0,
    processes: int = 4,
):
    """Run a training session: alternate fan-out self-play with a training step.

    `minutes` is the wall-clock budget for the whole session. It is split into
    generations so progress is checkpointed roughly every few minutes -- if the session
    dies halfway through, you lose at most one generation.
    """
    deadline = time.time() + minutes * 60
    generation_index = 0

    print(
        f"session: {minutes:.0f} min budget, {workers} self-play workers, "
        f"run={run_name}"
    )

    floor = minimum_generation_seconds(
        simulations or 200, parallel or 16, selfplay_fraction
    )
    print(f"  minimum generation length for these settings: {floor:.0f}s")

    while time.time() < deadline:
        remaining = deadline - time.time()
        # Below the floor a batch of games cannot finish, so the generation would cost
        # full price and produce nothing.
        if remaining < floor:
            print(f"  {remaining:.0f}s left -- below the {floor:.0f}s floor, stopping")
            break

        # Aim for several generations so progress checkpoints often, but never below
        # the floor.
        generation_seconds = min(
            remaining, MAX_GENERATION_SECONDS, max(floor, remaining / 3 + 60)
        )
        # Hard bound: a worker given more than this can outlive its own timeout, and a
        # killed worker loses every game it played.
        selfplay_seconds = min(
            generation_seconds * selfplay_fraction, max_safe_selfplay_seconds()
        )

        started = time.time()
        results = list(
            selfplay_worker.starmap(
                [
                    (
                        run_name,
                        worker_id,
                        selfplay_seconds,
                        games_per_worker,
                        simulations,
                        parallel,
                        processes,
                    )
                    for worker_id in range(workers)
                ]
            )
        )
        games = sum(r.get("games", 0) for r in results)
        positions = sum(r.get("positions", 0) for r in results)
        elapsed = time.time() - started
        print(
            f"  gen {generation_index}: {games} games / {positions} positions "
            f"from {len(results)} workers in {elapsed:.0f}s"
        )

        entry = train_generation.remote(
            run_name=run_name,
            new_games=games,
            new_positions=positions,
            selfplay_seconds=elapsed,
        )
        train = entry.get("train", {})
        print(
            f"  gen {generation_index}: loss {train.get('loss', 0):.3f} "
            f"(p {train.get('policy_loss', 0):.3f} v {train.get('value_loss', 0):.3f}) "
            f"over {entry.get('window_games', 0)} window games"
        )
        generation_index += 1

    print("\nsession complete. Check progress with:")
    print(f"  modal run modal_app.py::status --run-name {run_name}")
    print("Pull the newest checkpoint down for GUI play with:")
    print(f"  modal volume get {VOLUME_NAME} {run_name} ./runs/")


@app.local_entrypoint()
def detached(
    run_name: str = "chess",
    minutes: float = 180.0,
    workers: int = 10,
    games_per_worker: int = 200,
    simulations: int = 0,
    parallel: int = 0,
    processes: int = 4,
):
    """Launch a session that runs entirely on Modal, then exit immediately.

    Use this for long sessions: your machine is only needed for the few seconds it
    takes to start the job. Close the laptop, lose your wifi -- the session continues.

        modal run --detach modal_app.py::detached --minutes 180 --workers 10
    """
    call = run_session_remote.spawn(
        run_name=run_name,
        minutes=minutes,
        workers=workers,
        games_per_worker=games_per_worker,
        simulations=simulations,
        parallel=parallel,
        processes=processes,
    )
    print(f"session running remotely, call id: {call.object_id}")
    print(f"expected to finish in about {minutes:.0f} minutes")
    print("\nyou can close this terminal now. check progress any time with:")
    print(f"  modal run modal_app.py::status --run-name {run_name}")


@app.local_entrypoint()
def adjudicate(run_name: str = "chess", threshold: int = 5, relabel: bool = True):
    """Enable material adjudication of shuffled draws and relabel existing games."""
    import json

    print(json.dumps(enable_adjudication.remote(
        run_name=run_name, threshold=threshold, relabel=relabel
    ), indent=2))


@app.local_entrypoint()
def status(run_name: str = "chess", last: int = 15):
    import json

    print(json.dumps(run_status.remote(run_name=run_name, last=last), indent=2))


@app.local_entrypoint()
def evaluate(
    run_name: str = "chess",
    candidate: Optional[int] = None,
    baseline: Optional[int] = None,
    gap: int = 10,
    games: int = 60,
    simulations: int = 200,
    random_opening_plies: int = 10,
):
    result = evaluate_pair.remote(
        run_name=run_name,
        candidate=candidate,
        baseline=baseline,
        gap=gap,
        games=games,
        simulations=simulations,
        random_opening_plies=random_opening_plies,
    )
    print(result)

    decisive = result.get("wins", 0) + result.get("losses", 0)
    if result.get("games") and decisive < max(4, result["games"] * 0.2):
        print(
            f"\nwarning: only {decisive}/{result['games']} games were decisive, so this "
            f"Elo estimate carries almost no information. Compare checkpoints further "
            f"apart (--gap 20), play more games, or raise --random-opening-plies."
        )
