"""Package a checkpoint as a complete, installable UCI engine.

    python package_engine.py --run runs/chess_modal --generation 21
    python package_engine.py --run runs/chess_beam  --generation 14

Each call produces a self-contained engine you can install in Arena:

    engines/AstraZero_Gen21.bat     <- point the GUI at this
    engines/AstraZero_Gen21.bmp     <- logo, Arena's preferred format
    engines/AstraZero_Gen21.png     <- logo for GUIs that accept PNG
    engines/AstraZero_Gen21/        <- weights and config

Two things this handles that matter for engine-vs-engine matches:

* Each engine reports a DISTINCT name (AstraZero_Gen21, AstraZero_Gen14). Tournament
  managers key their cross-tables on the engine name, so two entrants sharing a name are
  merged into a single row and the match is unscorable.
* Optimizer state is stripped: two thirds of a training checkpoint, useless for playing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from az.core.checkpoint import CheckpointManager

LAUNCHER = """@echo off
rem ===========================================================================
rem  {name} - UCI chess engine
rem
rem  Arena:  Engines -> Install New Engine -> select THIS FILE -> choose UCI
rem  Works with Cute Chess, BanksiaGUI, Shredder and anything else UCI.
rem
rem  Trained on {games} self-play games. Generation {generation}.
rem ===========================================================================
rem
rem  @echo off is essential - anything on stdout that is not a UCI response
rem  corrupts the protocol and the GUI will hang or reject the engine.
rem
rem  cd /d "%~dp0..\\" is essential too - a GUI launches engines from its own
rem  directory, and the `az` package must be importable from the project root.
rem
rem  --simulations caps search depth; the GUI's clock usually stops it sooner.
rem  Put your own name in --author; it appears in tournament cross-tables.
rem ===========================================================================

cd /d "%~dp0..\\"
python uci.py --run "engines/{folder}" --name "{name}" --author "{author}" --simulations {simulations} %*
"""


def write_logo(source: Path, destination_stem: Path, size: int = 128) -> list[str]:
    """Render the logo at GUI icon size. Arena prefers BMP; others accept PNG."""
    try:
        from PIL import Image
    except ImportError:
        return []

    written = []
    with Image.open(source) as image:
        image = image.convert("RGB")

        # Trim the background, or the figure ends up tiny inside a square icon.
        #
        # Anchoring on bright pixels finds the figure reliably. Two things that do NOT
        # work on this artwork: a plain "darker or brighter than the background" mask
        # selects the dark vignette at the image edges and crops nothing, and a simple
        # brightness threshold stops above the wordmark, which is near-black. So take
        # the bright bounding box and extend downward to pick up a wordmark sitting
        # beneath the figure.
        grayscale = image.convert("L")
        box = grayscale.point(lambda value: 255 if value > 150 else 0).getbbox()
        if box:
            left, top, right, bottom = box
            width, height = right - left, bottom - top
            left = max(int(left - width * 0.08), 0)
            right = min(int(right + width * 0.08), image.width)
            top = max(int(top - height * 0.04), 0)
            bottom = min(int(bottom + height * 0.22), image.height)
            image = image.crop((left, top, right, bottom))

        image.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), (18, 18, 18))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        for suffix in (".bmp", ".png"):
            path = destination_stem.with_suffix(suffix)
            canvas.save(path)
            written.append(path.name)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Package an installable UCI engine")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--generation", type=int, help="default: newest")
    parser.add_argument("--out", type=Path, default=Path("engines"))
    parser.add_argument("--name", help="default: AstraZero_GenN")
    parser.add_argument("--author", default="AstraZero project")
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--logo", type=Path, default=Path("logo/logo.png"))
    args = parser.parse_args()

    manager = CheckpointManager(args.run)
    checkpoint = manager.load(args.generation, map_location="cpu")
    if checkpoint is None:
        print(f"no checkpoint in {args.run}. available: {manager.available_generations()}")
        return 1

    generation = checkpoint.run_state.generation
    name = args.name or f"AstraZero_Gen{generation}"
    folder = args.out / name
    target = folder / "checkpoints" / f"gen{generation:05d}"
    target.mkdir(parents=True, exist_ok=True)

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
        target / "model.pt",
    )

    (folder / "latest.json").write_text(
        json.dumps({"generation": generation, "path": f"gen{generation:05d}"}),
        encoding="utf-8",
    )
    source_config = args.run / "config.json"
    if source_config.exists():
        shutil.copy2(source_config, folder / "config.json")

    launcher = args.out / f"{name}.bat"
    launcher.write_text(
        LAUNCHER.format(
            name=name,
            folder=name,
            author=args.author,
            simulations=args.simulations,
            generation=generation,
            games=checkpoint.run_state.total_games,
        ),
        encoding="utf-8",
    )

    logos = write_logo(args.logo, args.out / name) if args.logo.exists() else []

    size_mb = (target / "model.pt").stat().st_size / 1e6
    print(f"packaged {name}")
    print(f"  launcher : {launcher}")
    print(f"  weights  : {target / 'model.pt'}  ({size_mb:.1f} MB)")
    print(f"  logo     : {', '.join(logos) if logos else 'skipped'}")
    print(f"  trained on {checkpoint.run_state.total_games} games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
