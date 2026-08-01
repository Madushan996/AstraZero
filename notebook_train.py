"""Train AstraZero in a Kaggle or Colab notebook, with the run surviving the session.

Notebooks are ephemeral: the container is wiped when the session ends. Training is not
ephemeral -- it is the accumulation of a replay buffer over weeks. This module bridges
the two by restoring the run at the start and saving it back at the end.

Kaggle (recommended: 4 CPU cores, 30 GPU-hours/week, headless 9-hour runs):

    !git clone https://github.com/Madushan996/AstraZero.git && cd AstraZero
    from notebook_train import session
    session(hours=8.5)

Colab (2 cores on the free tier, so roughly half the self-play throughput):

    from google.colab import drive; drive.mount('/content/drive')
    from notebook_train import session
    session(hours=3, store='/content/drive/MyDrive/astrazero')

Why the CPU count matters more than the GPU here: MCTS tree descent is single-threaded
Python, so throughput scales with cores, not with GPU class. A Kaggle notebook does about
twice the self-play of a free Colab one for that reason alone.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional


def detect_environment() -> str:
    if Path("/kaggle/working").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"


def default_store(environment: str) -> Path:
    """Where the run is kept BETWEEN sessions."""
    if environment == "kaggle":
        # Written to the notebook's output, which you save as a Dataset and attach to
        # the next session as /kaggle/input/<name>.
        return Path("/kaggle/working/astrazero")
    if environment == "colab":
        return Path("/content/drive/MyDrive/astrazero")
    return Path("runs")


def find_previous_run(environment: str, store: Path, run_name: str) -> Optional[Path]:
    """Locate a run saved by an earlier session, if there is one."""
    candidate = store / run_name
    if (candidate / "config.json").exists():
        return candidate

    if environment == "kaggle":
        # Attached datasets are read-only, so the run is copied out before use.
        for attached in sorted(Path("/kaggle/input").glob("*")):
            found = attached / run_name
            if (found / "config.json").exists():
                return found
    return None


def restore(environment: str, store: Path, run_name: str, work: Path) -> bool:
    """Copy a previous run into the working directory. Returns True if one was found."""
    previous = find_previous_run(environment, store, run_name)
    if previous is None:
        return False

    if previous.resolve() == work.resolve():
        return True

    work.parent.mkdir(parents=True, exist_ok=True)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(previous, work)

    shards = len(list((work / "games").glob("*.jsonl.gz")))
    print(f"restored {previous} -> {work} ({shards} game shards)")
    return True


def persist(work: Path, store: Path, run_name: str, keep_checkpoints: int = 2) -> None:
    """Save the run back, trimming checkpoints so the output stays small.

    Games are the irreplaceable part -- a few KB each and the whole history of what the
    engine has ever played. Checkpoints are ~145 MB each and regenerable by training on
    the games, so only the newest couple are worth carrying between sessions.
    """
    target = store / run_name
    target.mkdir(parents=True, exist_ok=True)

    for name in ("config.json", "latest.json", "history.jsonl"):
        source = work / name
        if source.exists():
            shutil.copy2(source, target / name)

    games = target / "games"
    games.mkdir(exist_ok=True)
    copied = 0
    for shard in (work / "games").glob("*.jsonl.gz"):
        destination = games / shard.name
        if not destination.exists():
            shutil.copy2(shard, destination)
            copied += 1

    checkpoints = target / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    generations = sorted((work / "checkpoints").glob("gen*"))
    for directory in generations[-keep_checkpoints:]:
        destination = checkpoints / directory.name
        if not destination.exists():
            shutil.copytree(directory, destination)

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(
        f"saved -> {target}  ({copied} new shards, "
        f"{len(generations[-keep_checkpoints:])} checkpoint(s), {size / 1e6:.0f} MB)"
    )


def session(
    hours: float = 8.0,
    run_name: str = "chess",
    store: Optional[str] = None,
    profile: str = "balanced",
    simulations: int = 200,
    parallel: int = 16,
    processes: int = 0,
    reserve_minutes: float = 12.0,
) -> None:
    """Restore, train for `hours`, save back.

    `reserve_minutes` is held back from the notebook's time limit so the run can be
    written out before the container is killed. Kaggle stops a session hard at its
    limit; losing the save would cost the whole session.
    """
    from az.core.pipeline import TrainingSession, default_config

    environment = detect_environment()
    store_path = Path(store) if store else default_store(environment)
    work = Path("/kaggle/working/run" if environment == "kaggle" else "runs") / run_name

    cores = os.cpu_count() or 1
    processes = processes or max(1, min(cores, 8))
    print(f"environment: {environment} | {cores} CPU core(s) | "
          f"{processes} self-play process(es)")

    resumed = restore(environment, store_path, run_name, work)
    config = None
    if not resumed:
        config = default_config("chess", profile=profile)
        config.selfplay["num_simulations"] = simulations
        config.selfplay["parallel_games"] = parallel
        config.game_kwargs["adjudicate_material_at"] = 5
        print("no previous run found -- starting a new one")

    training = TrainingSession(work, config=config)
    training.config.selfplay_processes = processes
    print(f"generation {training.run_state.generation}, "
          f"{training.run_state.total_games} games so far")

    minutes = max(1.0, hours * 60 - reserve_minutes)
    started = time.time()
    try:
        training.run_session(minutes=minutes)
    except KeyboardInterrupt:
        print("interrupted -- saving what completed")
    finally:
        # Always persist. A session that trained for eight hours and saved nothing is
        # worse than one that never ran.
        persist(work, store_path, run_name)
        print(f"elapsed {(time.time() - started) / 60:.0f} min")
        if environment == "kaggle":
            print(
                "\nNext: 'Save Version' -> the output becomes a Dataset. Attach that "
                "dataset to your next session and it resumes automatically."
            )
