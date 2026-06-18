#!/usr/bin/env python3
"""Summarize per-run stall metrics from mpv display logs.

A stall is any inter-frame interval exceeding --threshold (default 100ms).
Output: one row per (condition, strategy, timeout, run) with aggregated metrics.

Usage: python summarize-stalls.py <results-dir> [--threshold 100] [--fps 30]
"""

import csv
import sys
from pathlib import Path


def analyze_display_log(path, fps, threshold_ms):
    expected_ft = 1000.0 / fps
    frametimes = []
    with open(path) as f:
        for r in csv.DictReader(f):
            frametimes.append(float(r["frametime_ms"]))

    if not frametimes:
        return None

    total_playback_ms = sum(frametimes)
    num_frames = len(frametimes)
    video_duration_ms = (num_frames / fps) * 1000.0

    stalls = [ft for ft in frametimes if ft > threshold_ms]
    stall_excess = [ft - expected_ft for ft in stalls]

    return {
        "num_frames": num_frames,
        "stall_count": len(stalls),
        "stall_total_s": round(sum(stall_excess) / 1000.0, 3),
        "stall_from_duration_s": round(
            max(0, (total_playback_ms - video_duration_ms) / 1000.0), 3
        ),
        "stall_max_ms": round(max(stalls, default=0), 1),
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results-dir> [--threshold 100] [--fps 30]",
              file=sys.stderr)
        sys.exit(1)

    basedir = Path(sys.argv[1])
    fps = 30
    threshold_ms = 100

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

        analysis = analyze_display_log(display_path, fps, threshold_ms)
        if analysis is None:
            continue

        rows.append({
            "condition": condition,
            "strategy": strategy,
            "timeout_ms": timeout_ms,
            "run": run_num,
            **analysis,
        })

    rows.sort(key=lambda r: (r["condition"], r["strategy"], r["timeout_ms"], r["run"]))

    out_path = basedir / "stall_summary.csv"
    fieldnames = ["condition", "strategy", "timeout_ms", "run",
                  "num_frames", "stall_count", "stall_total_s",
                  "stall_from_duration_s", "stall_max_ms"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
