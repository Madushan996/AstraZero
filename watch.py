"""Watch a running Modal training session and shout if it stalls.

    python watch.py                          # sensible defaults
    python watch.py --interval 120           # poll every 2 minutes
    python watch.py --once                   # single status line, then exit

Why this exists: a self-play worker writes nothing until its whole generation finishes,
so "no output for 25 minutes" looks identical to "silently dead". That ambiguity already
cost an hour of compute once. This distinguishes them by watching two independent
signals -- containers alive, and shards landing -- and says plainly which one broke.

Only the Modal control plane is queried (app list, volume ls). No GPUs, no cost.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime

APP_PATTERN = re.compile(r"(ap-\w+)")


def run(command: list[str], timeout: int = 120) -> str:
    try:
        finished = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            # Modal draws tables with box characters; the console codepage mangles them
            # and silently broke column parsing.
            encoding="utf-8",
            errors="replace",
        )
        return (finished.stdout or "") + (finished.stderr or "")
    except (subprocess.TimeoutExpired, OSError) as error:
        return f"__ERROR__ {error}"


def columns(line: str) -> list[str]:
    """Split a Modal table row into cells, whatever the box characters decoded to."""
    cleaned = re.sub(r"[^\x20-\x7e]", "|", line)
    return [cell.strip() for cell in cleaned.split("|") if cell.strip()]


def running_app(app_name: str) -> tuple[str | None, int]:
    """Return (app_id, task_count) for the live ephemeral app, if any."""
    output = run(["modal", "app", "list"])
    if output.startswith("__ERROR__"):
        return None, -1

    lines = output.splitlines()
    for index, line in enumerate(lines):
        if app_name not in line:
            continue
        window = " ".join(lines[index : index + 3])
        if "ephemeral" not in window and "deployed" not in window:
            continue
        if "stopped" in line:
            continue
        app_id = APP_PATTERN.search(line)
        cells = columns(line)
        tasks = next((int(cell) for cell in cells if cell.isdigit()), -1)
        return (app_id.group(1) if app_id else None), tasks
    return None, 0


def shard_counts(volume: str, run_name: str) -> dict[str, int]:
    output = run(["modal", "volume", "ls", volume, f"{run_name}/games"])
    if output.startswith("__ERROR__"):
        return {}
    counts: dict[str, int] = {}
    for match in re.finditer(r"gen(\d{5})_", output):
        counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    return counts


def latest_generation(volume: str, run_name: str) -> str | None:
    output = run(["modal", "volume", "ls", volume, f"{run_name}/checkpoints"])
    if output.startswith("__ERROR__"):
        return None
    generations = sorted(re.findall(r"gen(\d{5})", output))
    return generations[-1] if generations else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a Modal training session")
    parser.add_argument("--run-name", default="chess2")
    parser.add_argument("--volume", default="alphazero-runs")
    parser.add_argument("--app-name", default="alphazero")
    parser.add_argument("--interval", type=int, default=180, help="seconds between polls")
    parser.add_argument("--max-minutes", type=float, default=200.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--stall-minutes",
        type=float,
        default=45.0,
        help="warn if no new shard appears for this long (a generation is ~25 min)",
    )
    args = parser.parse_args()

    started = time.time()
    deadline = started + args.max_minutes * 60
    last_total = -1
    last_change = time.time()

    print(f"watching {args.run_name} (poll every {args.interval}s, Ctrl+C to stop)\n")

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        app_id, tasks = running_app(args.app_name)
        counts = shard_counts(args.volume, args.run_name)
        total = sum(counts.values())
        newest = max(counts) if counts else "-----"
        # The current generation's own shard count is the live progress signal; the
        # running total barely moves and hides a stall.
        newest_count = counts.get(newest, 0)
        checkpoint = latest_generation(args.volume, args.run_name) or "?"
        elapsed = (time.time() - started) / 60

        if total != last_total:
            last_change = time.time()
            last_total = total
        quiet = (time.time() - last_change) / 60

        state = "running" if tasks > 0 else ("NO TASKS" if app_id else "not running")
        print(
            f"[{now}] {state:11s} tasks={tasks:<3} "
            f"gen{newest}: {newest_count:<3} shards (total {total:<5}) "
            f"checkpoint=gen{checkpoint} "
            f"quiet={quiet:4.1f}m  elapsed={elapsed:5.1f}m"
        )

        if args.once:
            return 0

        if app_id is None and elapsed > 2:
            print("\n=> session is no longer running. Check results with:")
            print(f"   modal run modal_app.py::status --run-name {args.run_name}")
            return 0

        if tasks == 0 and app_id is not None:
            # A finished session lingers in the listing with zero tasks for a while.
            # Recent progress distinguishes "just completed" from "died" -- without
            # this check a clean finish reports a false stall.
            if quiet < 5:
                print("\n=> no tasks left and progress just landed: session finished.")
                print(f"   modal run modal_app.py::status --run-name {args.run_name}")
                return 0
            print(f"\n!! app is listed with no tasks and no progress for {quiet:.0f}m.")
            print(f"   modal app logs {app_id}")
            return 1

        if quiet > args.stall_minutes:
            print(
                f"\n!! no new shards for {quiet:.0f} minutes (a generation is ~25). "
                f"Something is likely wrong."
            )
            print(f"   modal app logs {app_id}")
            return 1

        if time.time() >= deadline:
            print("\n=> watch window elapsed; session may still be running")
            return 0

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped watching (the session keeps running)")
        sys.exit(0)
