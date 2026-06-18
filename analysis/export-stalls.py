#!/usr/bin/env python3
"""Export per-stall data from mpv display logs to a CSV.

A stall is any inter-frame interval exceeding --threshold (default 100ms).
Output: one row per stall event with condition, strategy, timeout, run, duration.

Usage: python export-stalls.py <results-dir> [--threshold 100] [--fps 30]
"""

import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results-dir> [--threshold 100] [--fps 30]", file=sys.stderr)
        sys.exit(1)

    basedir = Path(sys.argv[1])
    threshold_ms = 100
    fps = 30

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--threshold":
            threshold_ms = int(args[i + 1])
            i += 2
        elif args[i] == "--fps":
            fps = int(args[i + 1])
            i += 2
        else:
            i += 1

    expected_ft = 1000.0 / fps
    out_path = basedir / "stalls.csv"
    rows = []

    for display_path in sorted(basedir.glob("*/*/run*/sub1_display.csv")):
        run_dir = display_path.parent
        condition = run_dir.parent.parent.name
        run_num = int(run_dir.name.removeprefix("run"))

        metrics_path = run_dir / "relay_metrics.csv"
        strategy = run_dir.parent.name
        timeout_ms = 0
        if metrics_path.exists():
            with open(metrics_path) as f:
                mr = list(csv.DictReader(f))
                if mr:
                    strategy = mr[-1]["strategy"]
                    timeout_ms = int(mr[-1]["timeout_ms"])

        with open(display_path) as f:
            for r in csv.DictReader(f):
                ft = float(r["frametime_ms"])
                if ft > threshold_ms:
                    rows.append({
                        "condition": condition,
                        "strategy": strategy,
                        "timeout_ms": timeout_ms,
                        "run": run_num,
                        "frame": int(r["display_seq"]),
                        "stall_ms": round(ft, 1),
                    })

    rows.sort(key=lambda r: (r["condition"], r["strategy"], r["timeout_ms"], r["run"], r["frame"]))

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "strategy", "timeout_ms", "run", "frame", "stall_ms"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} stalls to {out_path}")


if __name__ == "__main__":
    main()
