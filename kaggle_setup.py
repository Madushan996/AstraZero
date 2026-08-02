"""Stage the run for Kaggle and push a training notebook.

    python kaggle_setup.py stage        # build the upload folder + metadata
    python kaggle_setup.py upload       # create or version the dataset
    python kaggle_setup.py notebook     # push and run the training notebook

Requires Kaggle API credentials at ~/.kaggle/kaggle.json (Kaggle -> Settings -> API ->
Create New Token). Nothing here reads or transmits the token itself; the Kaggle client
does that.

The dataset is created PRIVATE. It is working data for one project, not something to
publish, and a private dataset attaches to your notebooks exactly the same way.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

STAGE = Path("kaggle_upload")
NOTEBOOK_DIR = Path("kaggle_notebook")

# Extension Kaggle's dataset pipeline leaves alone. Must match notebook_train.
KAGGLE_SAFE_SUFFIX = ".shard"


def stage(run_dir: Path, slug: str, username: str, keep_checkpoints: int = 1) -> None:
    """Copy the run into an upload folder, trimming to what a resume actually needs."""
    target = STAGE / "astrazero"
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (target / "games").mkdir(parents=True)

    for name in ("config.json", "latest.json", "history.jsonl"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, target / name)

    # Kaggle's dataset pipeline auto-decompresses anything it recognises as an archive.
    # Uploaded as .jsonl.gz, every shard was gunzipped into a directory containing a
    # truncated ".partial" file -- the whole replay buffer arrived corrupt. Renaming to
    # an extension Kaggle ignores gets the bytes through untouched; notebook_train
    # renames them back when restoring.
    shards = sorted((run_dir / "games").glob("*.jsonl.gz"))
    for shard in shards:
        safe = shard.name.replace(".jsonl.gz", KAGGLE_SAFE_SUFFIX)
        shutil.copy2(shard, target / "games" / safe)

    # Only the newest checkpoint travels. Older ones are ~145 MB each and the network
    # can be retrained from the games; the games cannot be recovered from anything.
    generations = sorted((run_dir / "checkpoints").glob("gen*"))
    for directory in generations[-keep_checkpoints:]:
        shutil.copytree(directory, target / "checkpoints" / directory.name)

    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "AstraZero training run",
                "id": f"{username}/{slug}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    size = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"staged -> {STAGE}")
    print(f"  {len(shards)} game shards")
    print(f"  {len(generations[-keep_checkpoints:])} checkpoint(s)")
    print(f"  {size / 1e6:.0f} MB total")


def upload(slug: str, username: str, message: str) -> None:
    """Create the dataset, or push a new version if it already exists."""
    if not (STAGE / "dataset-metadata.json").exists():
        print("nothing staged; run 'stage' first")
        return

    existing = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "status", f"{username}/{slug}"],
        capture_output=True, text=True,
    )
    known = existing.returncode == 0 and "404" not in (existing.stdout + existing.stderr)

    command = (
        [sys.executable, "-m", "kaggle", "datasets", "version",
         "-p", str(STAGE), "-m", message, "--dir-mode", "zip"]
        if known else
        [sys.executable, "-m", "kaggle", "datasets", "create",
         "-p", str(STAGE), "--dir-mode", "zip"]
    )
    print(("versioning" if known else "creating") + f" dataset {username}/{slug}...")
    subprocess.run(command, check=False)


NOTEBOOK_SOURCE = """import subprocess, sys, os
subprocess.run(["pip", "install", "-q", "python-chess"], check=False)
subprocess.run(
    ["git", "clone", "--depth", "1",
     "https://github.com/{repo}.git", "/kaggle/working/AstraZero"],
    check=False,
)
sys.path.insert(0, "/kaggle/working/AstraZero")
os.chdir("/kaggle/working/AstraZero")

from notebook_train import session
session(hours={hours}, run_name="astrazero", simulations={simulations})
"""


def notebook(
    slug: str,
    username: str,
    dataset: str,
    repo: str,
    hours: float,
    accelerator: str = "gpu-t4x2",
    simulations: int = 200,
) -> None:
    """Write and push a notebook that trains and saves its output.

    `accelerator` matters more than it looks. Left to Kaggle's default, a session can be
    assigned a Tesla P100 (sm_60), which Kaggle's own PyTorch build does not support --
    the run then falls back to CPU while still consuming GPU quota. A T4 is sm_75 and
    works. Pass "none" to skip the accelerator entirely and keep the quota.
    """
    if NOTEBOOK_DIR.exists():
        shutil.rmtree(NOTEBOOK_DIR)
    NOTEBOOK_DIR.mkdir(parents=True)

    source = NOTEBOOK_SOURCE.format(
        repo=repo, hours=hours, simulations=simulations
    )
    cells = [{
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": source.splitlines(keepends=True),
    }]
    (NOTEBOOK_DIR / "astrazero-train.ipynb").write_text(
        json.dumps({
            "cells": cells,
            "metadata": {"kernelspec": {
                "name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 4,
        }, indent=1),
        encoding="utf-8",
    )

    metadata = {
        "id": f"{username}/{slug}",
        "title": "AstraZero training",
        "code_file": "astrazero-train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": accelerator != "none",
        "enable_internet": True,  # needed to clone the repo and pip install
        "dataset_sources": [dataset],
        "competition_sources": [],
        "kernel_sources": [],
    }
    if accelerator != "none":
        metadata["machine_shape"] = accelerator

    (NOTEBOOK_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    command = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(NOTEBOOK_DIR)]
    if accelerator != "none":
        command += ["--accelerator", accelerator]

    print(f"pushing notebook {username}/{slug} "
          f"(dataset: {dataset}, accelerator: {accelerator})...")
    subprocess.run(command, check=False)
    print(f"\nwatch it at https://www.kaggle.com/code/{username}/{slug}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle staging and launch")
    parser.add_argument("command", choices=["stage", "upload", "notebook"])
    parser.add_argument("--run", type=Path, default=Path("runs/astrazero"))
    parser.add_argument("--username", default="madushan996")
    parser.add_argument("--dataset-slug", default="astrazero-run")
    parser.add_argument("--notebook-slug", default="astrazero-training")
    parser.add_argument("--repo", default="Madushan996/AstraZero")
    parser.add_argument("--hours", type=float, default=8.5)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--message", default="new generation")
    parser.add_argument(
        "--accelerator",
        default="gpu-t4x2",
        help=(
            "gpu-t4x2 (works), gpu-p100 (unsupported by Kaggle's PyTorch), "
            "or none to run on CPU without spending GPU quota"
        ),
    )
    args = parser.parse_args()

    if args.command == "stage":
        stage(args.run, args.dataset_slug, args.username)
    elif args.command == "upload":
        upload(args.dataset_slug, args.username, args.message)
    else:
        notebook(
            args.notebook_slug, args.username,
            f"{args.username}/{args.dataset_slug}", args.repo, args.hours,
            accelerator=args.accelerator,
            simulations=args.simulations,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
