#!/usr/bin/env python3
"""Analyse decodable video time, stalls, and furthest progress from latency data.

A frame is decodable iff:
  1. The I-frame (object 0) for its GoP was received, AND
  2. All P-frames with object_id < this frame's object_id were also received
     (contiguous run from object 0).

Metrics per (condition, strategy, timeout), averaged across runs:
  - decodable_s:  total decodable video duration (seconds)
  - total_s:      total video duration
  - stall_count:  number of stall events (consecutive non-decodable GoPs)
  - stall_total_s: total stall duration
  - stall_max_s:  longest single stall
  - progress_s:   PTS of last decodable frame (furthest point reached)
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


FPS = 30.0


def analyse(latency_path: Path):
    runs = defaultdict(lambda: defaultdict(set))
    with open(latency_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["condition"], row["strategy"], int(row["timeout_ms"]), int(row["run"]))
            runs[key][int(row["group_id"])].add(int(row["object_id"]))

    # GoP sizes from baseline/none (ground truth - receives everything)
    gop_sizes = {}
    for key, groups in runs.items():
        cond, strat, tmo, run = key
        if cond == "baseline" and strat == "none":
            for gid, objs in groups.items():
                gop_sizes[(run, gid)] = max(objs) + 1

    results = defaultdict(list)

    for key, groups in runs.items():
        cond, strat, tmo, run = key

        all_group_ids = sorted(gid for (r, gid) in gop_sizes if r == run)
        if not all_group_ids:
            continue

        gop_duration = {}
        for gid in all_group_ids:
            gop_duration[gid] = gop_sizes.get((run, gid), 0) / FPS

        total_s = sum(gop_duration[g] for g in all_group_ids)
        total_frames = sum(gop_sizes.get((run, g), 0) for g in all_group_ids)

        # Per-GoP: count decodable frames (contiguous from object 0)
        decodable_per_gop = {}
        for gid in all_group_ids:
            if gid not in groups or 0 not in groups[gid]:
                decodable_per_gop[gid] = 0
                continue
            objs = groups[gid]
            contiguous = 0
            for oid in range(gop_sizes.get((run, gid), 0)):
                if oid in objs:
                    contiguous += 1
                else:
                    break
            decodable_per_gop[gid] = contiguous

        decodable_s = sum(decodable_per_gop[g] / FPS for g in all_group_ids)
        decodable_frames = sum(decodable_per_gop[g] for g in all_group_ids)

        # Stall analysis: a stall is a run of consecutive GoPs with 0 decodable frames
        stalls = []
        current_stall = 0.0
        for gid in all_group_ids:
            if decodable_per_gop[gid] == 0:
                current_stall += gop_duration[gid]
            else:
                if current_stall > 0:
                    stalls.append(current_stall)
                    current_stall = 0.0
        if current_stall > 0:
            stalls.append(current_stall)

        progress_s = 0.0
        cumulative_s = 0.0
        for gid in all_group_ids:
            if decodable_per_gop[gid] > 0:
                progress_s = cumulative_s + decodable_per_gop[gid] / FPS
            cumulative_s += gop_duration[gid]

        results[(cond, strat, tmo)].append({
            "total_s": total_s,
            "total_frames": total_frames,
            "decodable_s": decodable_s,
            "decodable_frames": decodable_frames,
            "decodable_ratio": decodable_frames / total_frames if total_frames else 0,
            "stall_count": len(stalls),
            "stall_total_s": sum(stalls),
            "stall_max_s": max(stalls) if stalls else 0.0,
            "progress_s": progress_s,
        })

    print("=== Decodable Video Time ===")
    print()
    print(f"{'CONDITION':<12} {'STRATEGY':<20} {'TMO':>6} {'RUNS':>4} | "
          f"{'DEC_S':>6} {'TOT_S':>5} {'DEC_R':>6} | "
          f"{'PROG_S':>6} {'PROG%':>6}")
    print(f"{'─'*12} {'─'*20} {'─'*6} {'─'*4}─┼─"
          f"{'─'*6}─{'─'*5}─{'─'*6}─┼─"
          f"{'─'*6}─{'─'*6}")

    for key in sorted(results.keys()):
        rl = results[key]
        n = len(rl)
        cond, strat, tmo = key
        tmo_str = "-" if tmo == 0 else f"{tmo}ms"
        a = lambda f: sum(r[f] for r in rl) / n

        print(f"{cond:<12} {strat:<20} {tmo_str:>6} {n:>4} | "
              f"{a('decodable_s'):>6.1f} {a('total_s'):>5.1f} {a('decodable_ratio'):>6.3f} | "
              f"{a('progress_s'):>6.1f} {a('progress_s')/a('total_s')*100 if a('total_s') else 0:>5.1f}%")

    print()
    print("=== Stall Events (non-decodable GoP gaps) ===")
    print()
    print(f"{'CONDITION':<12} {'STRATEGY':<20} {'TMO':>6} {'RUNS':>4} | "
          f"{'STALLS':>6} {'TOT_S':>7} {'MAX_S':>7} {'AVG_S':>7}")
    print(f"{'─'*12} {'─'*20} {'─'*6} {'─'*4}─┼─"
          f"{'─'*6}─{'─'*7}─{'─'*7}─{'─'*7}")

    for key in sorted(results.keys()):
        rl = results[key]
        n = len(rl)
        cond, strat, tmo = key
        tmo_str = "-" if tmo == 0 else f"{tmo}ms"
        a = lambda f: sum(r[f] for r in rl) / n

        avg_stall_s = a('stall_total_s') / a('stall_count') if a('stall_count') > 0 else 0

        print(f"{cond:<12} {strat:<20} {tmo_str:>6} {n:>4} | "
              f"{a('stall_count'):>6.1f} {a('stall_total_s'):>7.2f} "
              f"{a('stall_max_s'):>7.2f} {avg_stall_s:>7.2f}")

    csv_path = latency_path.parent / "decodability.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "strategy", "timeout_ms", "runs",
                     "avg_decodable_s", "avg_total_s", "avg_decodable_ratio",
                     "avg_stall_count", "avg_stall_total_s", "avg_stall_max_s",
                     "avg_progress_s"])
        for key in sorted(results.keys()):
            rl = results[key]
            n = len(rl)
            cond, strat, tmo = key
            a = lambda f: sum(r[f] for r in rl) / n
            w.writerow([cond, strat, tmo, n,
                        f"{a('decodable_s'):.2f}",
                        f"{a('total_s'):.2f}",
                        f"{a('decodable_ratio'):.4f}",
                        f"{a('stall_count'):.1f}",
                        f"{a('stall_total_s'):.2f}",
                        f"{a('stall_max_s'):.2f}",
                        f"{a('progress_s'):.2f}"])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/paper-data/latency.csv")
    analyse(path)
