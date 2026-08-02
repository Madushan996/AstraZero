"""Watch a Kaggle training notebook from the terminal.

    python watch_kaggle.py                 # poll until it finishes
    python watch_kaggle.py --once          # single status line

Kaggle's own page shows live logs, so this exists for two things it does not do well:
tell you at a glance whether the run is healthy, and shout if it silently started from
scratch.

That last one matters more than it sounds. If the replay buffer is not found, the session
does not fail -- it happily trains a brand-new network for eight hours and produces
something worse than you started with. It looks identical to a successful run until you
read the logs. This watches for it explicitly.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

# Lines that mean the run is doing the right thing.
HEALTHY = ("restored ", "[resume] generation", "unpacking ")
# Lines that mean it is training from nothing, having failed to find the buffer.
ALARM = ("no previous run found", "[new run]")
DONE = ("COMPLETE", "ERROR", "CANCEL")


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    parser = argparse.ArgumentParser(description="Watch a Kaggle notebook run")
    parser.add_argument("--kernel", default="madushanb/astrazero-training")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=600.0)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    started = time.time()
    seen = 0
    verdict = None

    print(f"watching {args.kernel} (Ctrl+C to stop; the run continues)\n")

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        elapsed = (time.time() - started) / 60

        try:
            status = str(getattr(api.kernels_status(args.kernel), "status", "?"))
        except Exception as error:
            status = f"status unavailable ({type(error).__name__})"

        try:
            logs = api.kernels_logs(args.kernel) or ""
        except Exception:
            logs = ""

        # Only print what is new, so a long poll does not repeat the whole log.
        fresh = logs[seen:]
        seen = len(logs)

        print(f"[{now}] {status:34s} log {len(logs):>7} chars  elapsed {elapsed:5.1f}m")

        for line in (l.strip() for l in fresh.splitlines() if l.strip()):
            interesting = any(k in line for k in HEALTHY + ALARM)
            if interesting or "games" in line or "loss" in line:
                print(f"           | {line[:150]}")

        if verdict is None and logs:
            if any(k in logs for k in ALARM):
                verdict = "alarm"
                print(
                    "\n!! This run did NOT find the previous buffer and is training a "
                    "fresh network.\n   Stop it: python -c \"from kaggle.api."
                    "kaggle_api_extended import KaggleApi; ...\" or cancel on the "
                    "Kaggle page.\n"
                )
            elif any(k in logs for k in HEALTHY):
                verdict = "healthy"
                print("           => buffer restored; this run is continuing the run\n")

        if args.once:
            return 0
        if any(state in status.upper() for state in DONE):
            print(f"\nfinished with status {status}")
            print("Save Version on the notebook page turns the output into a dataset "
                  "you can attach to the next session.")
            return 0
        if elapsed > args.max_minutes:
            print("\nwatch window elapsed; the run may still be going")
            return 0

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped watching (the Kaggle run keeps going)")
        sys.exit(0)
