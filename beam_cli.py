"""Invoke the Beam functions in beam_app.py.

`beam run` in this SDK launches a container from a Pod abstraction; functions declared
with `@function` are instead invoked from Python via `.remote()`. This is the thin
wrapper that does that.

    python beam_cli.py seed
    python beam_cli.py init
    python beam_cli.py status
    python beam_cli.py eval --baseline 0 --games 40
    python beam_cli.py export
"""

from __future__ import annotations

import argparse
import json


def _download(beam_app, args) -> dict:
    """Stream a checkpoint back from the volume and write it locally.

    Beam has no volume download, so the file comes back through repeated function calls.
    The sha256 from prepare_export is verified at the end -- a checkpoint that is
    silently truncated would load and then play badly, which is much worse than failing.
    """
    import base64
    import hashlib
    import json as _json
    from pathlib import Path

    prepared = beam_app.prepare_export.remote(
        run_name=args.run_name, generation=args.generation
    )
    if prepared.get("status") != "ready":
        return prepared

    generation = prepared["generation"]
    total = prepared["bytes"]
    chunk = max(100_000, int(args.chunk_mb * 1_000_000))
    print(f"downloading generation {generation}: {prepared['mb']} MB")

    blocks: list[bytes] = []
    offset = 0
    while offset < total:
        response = beam_app.fetch_export_chunk.remote(
            run_name=args.run_name,
            generation=generation,
            offset=offset,
            length=chunk,
        )
        if response.get("status") != "ok":
            return {"status": "chunk fetch failed", "at": offset, "detail": response}

        block = base64.b64decode(response["data"])
        if not block:
            return {"status": "empty chunk", "at": offset}
        blocks.append(block)
        offset += len(block)
        print(f"  {offset / 1e6:6.1f} / {total / 1e6:.1f} MB", flush=True)

    data = b"".join(blocks)
    digest = hashlib.sha256(data).hexdigest()
    if digest != prepared["sha256"]:
        return {"status": "checksum mismatch -- discarded", "expected": prepared["sha256"]}

    out = Path(args.out)
    (out / "checkpoints" / f"gen{generation:05d}").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints" / f"gen{generation:05d}" / "model.pt").write_bytes(data)
    (out / "latest.json").write_text(
        _json.dumps({"generation": generation, "path": f"gen{generation:05d}"}),
        encoding="utf-8",
    )

    return {
        "status": "downloaded",
        "generation": generation,
        "mb": prepared["mb"],
        "run_dir": str(out),
        "play_with": f"python uci.py --run {out}",
    }


def _download_games(beam_app, args) -> dict:
    """Stream the whole replay buffer back and unpack it locally."""
    import base64
    import hashlib
    import tarfile
    from pathlib import Path

    prepared = beam_app.prepare_games_bundle.remote(run_name=args.run_name)
    if prepared.get("status") != "ready":
        return prepared

    total = prepared["bytes"]
    chunk = max(100_000, int(args.chunk_mb * 1_000_000))
    print(f"downloading {prepared['shards']} shards, {prepared['mb']} MB")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "games.tar"

    digest = hashlib.sha256()
    offset = 0
    with archive_path.open("wb") as handle:
        while offset < total:
            response = beam_app.fetch_file_chunk.remote(
                run_name=args.run_name,
                name=prepared["name"],
                offset=offset,
                length=chunk,
            )
            if response.get("status") != "ok":
                return {"status": "chunk failed", "at": offset, "detail": response}
            block = base64.b64decode(response["data"])
            if not block:
                return {"status": "empty chunk", "at": offset}
            handle.write(block)
            digest.update(block)
            offset += len(block)
            print(f"  {offset / 1e6:6.1f} / {total / 1e6:.1f} MB", flush=True)

    if digest.hexdigest() != prepared["sha256"]:
        archive_path.unlink(missing_ok=True)
        return {"status": "checksum mismatch -- discarded"}

    with tarfile.open(archive_path) as tar:
        tar.extractall(out)
    archive_path.unlink(missing_ok=True)

    if prepared.get("config"):
        (out / "config.json").write_text(prepared["config"], encoding="utf-8")

    shards = len(list((out / "games").glob("*.jsonl.gz")))
    return {
        "status": "downloaded",
        "shards": shards,
        "mb": prepared["mb"],
        "out": str(out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Beam control for the AlphaZero run")
    parser.add_argument("--run-name", default="chess")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="copy beam_seed/ games and config onto the volume")

    init = sub.add_parser("init", help="create the run and a generation-0 checkpoint")
    init.add_argument("--game", default="chess")
    init.add_argument("--profile", default="balanced")

    status = sub.add_parser("status", help="show run progress")
    status.add_argument("--last", type=int, default=10)

    evaluate = sub.add_parser("eval", help="play two checkpoints head to head")
    evaluate.add_argument("--candidate", type=int)
    evaluate.add_argument("--baseline", type=int)
    evaluate.add_argument("--gap", type=int, default=10)
    evaluate.add_argument("--games", type=int, default=60)
    evaluate.add_argument("--simulations", type=int, default=200)
    evaluate.add_argument("--random-opening-plies", type=int, default=10)

    export = sub.add_parser("export", help="locate the newest checkpoint")
    export.add_argument("--generation", type=int)

    download = sub.add_parser(
        "download", help="fetch a checkpoint to this machine for GUI play"
    )
    download.add_argument("--generation", type=int, help="default: newest")
    download.add_argument("--out", default="runs/chess", help="local run directory")
    download.add_argument(
        "--chunk-mb",
        type=float,
        default=4.0,
        help="lower this if a chunk exceeds the gRPC message limit",
    )

    games = sub.add_parser(
        "download-games", help="fetch the whole replay buffer (for platform migration)"
    )
    games.add_argument("--out", default="runs/chess_beam")
    games.add_argument("--chunk-mb", type=float, default=4.0)

    session = sub.add_parser(
        "session", help="run a full session server-side (laptop can disconnect)"
    )
    session.add_argument("--minutes", type=float, default=120.0)
    session.add_argument("--workers", type=int, default=6)
    session.add_argument("--processes", type=int, default=4)
    session.add_argument("--games-per-worker", type=int, default=400)
    session.add_argument("--simulations", type=int, default=0)
    session.add_argument("--parallel", type=int, default=0)

    args = parser.parse_args()

    # Imported here so `--help` works without contacting Beam.
    import beam_app

    if args.command == "seed":
        result = beam_app.seed_volume.remote(run_name=args.run_name)
    elif args.command == "init":
        result = beam_app.init_run.remote(
            run_name=args.run_name, game=args.game, profile=args.profile
        )
    elif args.command == "status":
        result = beam_app.run_status.remote(run_name=args.run_name, last=args.last)
    elif args.command == "eval":
        result = beam_app.evaluate_pair.remote(
            run_name=args.run_name,
            candidate=args.candidate,
            baseline=args.baseline,
            gap=args.gap,
            games=args.games,
            simulations=args.simulations,
            random_opening_plies=args.random_opening_plies,
        )
        if isinstance(result, dict) and result.get("games"):
            decisive = result.get("wins", 0) + result.get("losses", 0)
            if decisive < max(4, result["games"] * 0.2):
                print(
                    f"\nwarning: only {decisive}/{result['games']} games were decisive, "
                    f"so this Elo estimate carries almost no information. Compare "
                    f"checkpoints further apart, play more games, or raise "
                    f"--random-opening-plies."
                )
    elif args.command == "export":
        result = beam_app.export_checkpoint.remote(
            run_name=args.run_name, generation=args.generation
        )
    elif args.command == "download":
        result = _download(beam_app, args)
    elif args.command == "download-games":
        result = _download_games(beam_app, args)
    elif args.command == "session":
        print(
            f"launching a {args.minutes:.0f}-minute session server-side. The loop runs "
            f"on Beam, so a dropped connection will not stop it -- recover the summary "
            f"with 'python beam_cli.py status'."
        )
        result = beam_app.run_session_remote.remote(
            run_name=args.run_name,
            minutes=args.minutes,
            workers=args.workers,
            games_per_worker=args.games_per_worker,
            simulations=args.simulations,
            parallel=args.parallel,
            processes=args.processes,
        )
    else:  # pragma: no cover - argparse enforces this
        parser.error(f"unknown command {args.command}")

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
