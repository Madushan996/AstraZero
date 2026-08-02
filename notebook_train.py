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

# Extension the run's shards are stored under on Kaggle. Kaggle's dataset pipeline
# gunzips anything ending in .gz, which corrupted the entire replay buffer the first
# time this ran. Must match kaggle_setup.py.
KAGGLE_SAFE_SUFFIX = ".shard"


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
    """Locate a run saved by an earlier session, if there is one.

    Searches several layouts because a Kaggle dataset can be attached with the run at
    its root (/kaggle/input/my-run/config.json) or nested one level down
    (/kaggle/input/my-run/astrazero/config.json), depending on how it was uploaded.
    Guessing wrong means silently starting from scratch and throwing away the buffer,
    so check both rather than assume.
    """
    searched = [store / run_name, store]
    if environment == "kaggle":
        # Any depth: Kaggle's layout depends on how the dataset was uploaded, and
        # guessing wrong has already cost two sessions.
        for config in sorted(Path("/kaggle/input").rglob("config.json")):
            searched.append(config.parent)

    for candidate in searched:
        if (candidate / "config.json").exists() and (candidate / "games").is_dir():
            return candidate

    # Datasets uploaded with --dir-mode zip arrive as a single archive rather than an
    # extracted tree. Unpack it rather than reporting "no previous run" -- that failure
    # looks exactly like a fresh start and would silently discard the whole buffer.
    return _extract_archive(environment, run_name)


def _extract_archive(environment: str, run_name: str) -> Optional[Path]:
    import zipfile

    roots = [Path("/kaggle/input")] if environment == "kaggle" else []
    for root in roots:
        for archive in sorted(root.glob("*/*.zip")):
            destination = Path("/kaggle/working/_unpacked") / archive.stem
            if not destination.exists():
                print(f"unpacking {archive.name} ...")
                destination.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(destination)
            for candidate in (destination / run_name, destination):
                if (candidate / "config.json").exists() and (candidate / "games").is_dir():
                    return candidate
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

    # Shards are uploaded as ".shard" because Kaggle gunzips anything ending in .gz,
    # which silently corrupted the entire buffer the first time. Put the real extension
    # back so the replay buffer can find them.
    renamed = 0
    for shard in (work / "games").glob(f"*{KAGGLE_SAFE_SUFFIX}"):
        shard.rename(shard.with_name(shard.name.replace(KAGGLE_SAFE_SUFFIX, ".jsonl.gz")))
        renamed += 1

    shards = len(list((work / "games").glob("*.jsonl.gz")))
    print(f"restored {previous} -> {work} ({shards} game shards"
          + (f", {renamed} renamed)" if renamed else ")"))
    if shards == 0:
        raise RuntimeError(
            f"restored {previous} but found no readable shards. Refusing to continue: "
            f"training from an empty buffer would discard the run."
        )
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


# Measured seconds of one worker process per (ply x simulation x concurrent game),
# with the network on a GPU. CPU-only inference is roughly 2.5x slower.
SECONDS_PER_PLY_SIM_GAME = 0.0012
TYPICAL_PLIES = 180


def safe_parallel(simulations: int, window_seconds: float, on_gpu: bool) -> int:
    """Largest batch width whose games can actually finish inside a generation.

    A worker plays `parallel` games at once and they complete together, so a batch that
    outlasts the generation produces NOTHING while costing the full time. The stored
    config carries `parallel_games: 96` from a cloud profile; on a notebook CPU that is
    a ~10,000 second batch against a ~600 second window.
    """
    per_game = SECONDS_PER_PLY_SIM_GAME * TYPICAL_PLIES * simulations
    if not on_gpu:
        per_game *= 2.5
    # Two batches per generation, so a slow one still leaves something completed.
    return max(2, int(window_seconds / (per_game * 2)))


def _describe_inputs(limit: int = 25) -> str:
    """List what is actually mounted, so a failed lookup is diagnosable from the log."""
    root = Path("/kaggle/input")
    if not root.exists():
        return f"  {root} does not exist (is a dataset attached?)"

    lines = [f"  contents of {root}:"]
    entries = sorted(root.rglob("*"))
    if not entries:
        lines.append("    (empty -- no dataset attached to this notebook)")
    for path in entries[:limit]:
        kind = "DIR " if path.is_dir() else f"{path.stat().st_size:>10,}"
        lines.append(f"    {kind}  {path}")
    if len(entries) > limit:
        lines.append(f"    ... and {len(entries) - limit} more")
    return "\n".join(lines)


def session(
    hours: float = 8.0,
    run_name: str = "chess",
    store: Optional[str] = None,
    profile: str = "balanced",
    simulations: int = 200,
    parallel: int = 0,  # 0 = size it from the session length and device
    processes: int = 0,
    reserve_minutes: float = 12.0,
    min_games: int = 200,
    allow_new_run: bool = False,
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

    print(_describe_inputs())
    resumed = restore(environment, store_path, run_name, work)
    config = None
    if not resumed:
        if not allow_new_run:
            # Starting from scratch must be a decision, never a fallback. Twice now a
            # session trained a fresh network for its whole duration because discovery
            # failed silently, and the logs looked healthy the entire time.
            raise RuntimeError(
                "no previous run found, and allow_new_run is False.\n"
                + _describe_inputs()
                + "\nIf you really want to start from zero, pass allow_new_run=True."
            )
        config = default_config("chess", profile=profile)
        config.selfplay["num_simulations"] = simulations
        config.selfplay["parallel_games"] = parallel
        config.game_kwargs["adjudicate_material_at"] = 5
        print("starting a NEW run (allow_new_run=True)")

    training = TrainingSession(work, config=config)
    training.config.selfplay_processes = processes

    minutes = max(1.0, hours * 60 - reserve_minutes)

    # Search width and depth are per-session tuning, not part of the run definition, so
    # override whatever the stored config inherited from a cloud profile. Sizing this to
    # the actual machine is what stops a session producing zero games.
    window = (minutes * 60) / 4 * 0.8  # roughly one generation's self-play budget
    on_gpu = training.device.type == "cuda"
    chosen = parallel or safe_parallel(simulations, window, on_gpu)
    training.config.selfplay["num_simulations"] = simulations
    training.config.selfplay["parallel_games"] = chosen
    print(
        f"generation {training.run_state.generation}, "
        f"{training.run_state.total_games} games so far | "
        f"{simulations} sims x {chosen} parallel on {training.device.type}"
    )
    started = time.time()
    try:
        training.run_session(minutes=minutes, min_games=min_games)
    except KeyboardInterrupt:
        print("interrupted -- saving what completed")
    finally:
        # Always persist. A session that trained for eight hours and saved nothing is
        # worse than one that never ran.
        persist(work, store_path, run_name)

        if environment == "kaggle":
            # Everything under /kaggle/working becomes the version's output, so drop the
            # working copy and the unpacked archive -- both duplicate the store and would
            # triple the download when harvesting.
            for junk in (work.parent, Path("/kaggle/working/_unpacked")):
                if junk.exists() and store_path not in junk.parents and junk != store_path:
                    shutil.rmtree(junk, ignore_errors=True)

        print(f"elapsed {(time.time() - started) / 60:.0f} min")
        if environment == "kaggle":
            print(
                "\nThis run's output is saved automatically when the session COMPLETES.\n"
                "Do not click 'Save Version' -- that starts a separate run from the old\n"
                "dataset. Harvest instead:  python kaggle_setup.py harvest"
            )
