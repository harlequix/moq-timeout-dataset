#!/usr/bin/env python3
"""Per-run reset-abandonment metrics: where in a group each reset fired.

For every group reset on timeout (from relay.log), measures whether the
keyframe (object 0) had been delivered, what fraction of the group's objects
reached the subscriber, and the object index that triggered the reset.

Usage: python abandonment.py <results-dir>
"""

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TIMEOUT_RE = re.compile(r"timeout exceeded.*group=(\d+) object=(\d+)")


def analyze_run(run_dir):
    group_total = defaultdict(int)
    with open(run_dir / "manifest.csv") as f:
        for r in csv.DictReader(f):
            group_total[int(r["group_id"])] += 1

    recv = defaultdict(set)
    with open(run_dir / "sub1_recv.csv") as f:
        for r in csv.DictReader(f):
            recv[int(r["group_id"])].add(int(r["object_id"]))

    triggers = {}
    with open(run_dir / "relay.log") as f:
        for line in f:
            m = TIMEOUT_RE.search(line)
            if m:
                triggers.setdefault(int(m.group(1)), int(m.group(2)))

    reset_groups = len(triggers)
    kf_in_reset = sum(1 for g in triggers if 0 in recv.get(g, set()))
    fracs = [len(recv.get(g, set())) / group_total[g]
             for g in triggers if group_total.get(g)]

    return {
        "reset_groups": reset_groups,
        "complete_groups": len(group_total) - reset_groups,
        "kf_in_reset": kf_in_reset,
        "recv_frac_reset": round(statistics.mean(fracs), 4) if fracs else "",
        "trigger_obj_median": int(statistics.median(triggers.values())) if triggers else "",
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results-dir>", file=sys.stderr)
        sys.exit(1)

    basedir = Path(sys.argv[1])
    rows = []
    for recv_path in sorted(basedir.glob("*/*/run*/sub1_recv.csv")):
        run_dir = recv_path.parent
        condition = run_dir.parent.parent.name
        run_num = int(run_dir.name.removeprefix("run"))

        strategy = run_dir.parent.name
        timeout_ms = 0
        metrics_path = run_dir / "relay_metrics.csv"
        if metrics_path.exists():
            with open(metrics_path) as f:
                mr = list(csv.DictReader(f))
                if mr:
                    strategy = mr[-1]["strategy"]
                    timeout_ms = int(mr[-1]["timeout_ms"])

        rows.append({
            "condition": condition,
            "strategy": strategy,
            "timeout_ms": timeout_ms,
            "run": run_num,
            **analyze_run(run_dir),
        })

    rows.sort(key=lambda r: (r["condition"], r["strategy"], r["timeout_ms"], r["run"]))

    out_path = basedir / "abandonment.csv"
    fieldnames = ["condition", "strategy", "timeout_ms", "run",
                  "reset_groups", "complete_groups", "kf_in_reset",
                  "recv_frac_reset", "trigger_obj_median"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
