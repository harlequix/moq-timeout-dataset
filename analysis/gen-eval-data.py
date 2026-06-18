#!/usr/bin/env python3
"""Generate LaTeX tables, value macros, and statistical analysis from experiment results.

Usage: python gen-eval-data.py <results-dir> [<output-dir>]

Reads summary.csv, latency.csv, decodability.csv, mpv_stalls.csv from
<results-dir> and writes LaTeX files into <output-dir> (default: gen/).

Generated files:
  values.tex         - \\newcommand macros for individual data points
  tab_results.tex    - main results table (condition x strategy x timeout)
  tab_stalls.tex     - playback continuity table (decodable time, stalls)
  tab_compact.tex    - compact table with selected timeouts
  tab_stats.tex      - statistical significance table (p-values)
  stats_report.txt   - human-readable statistical summary
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt_int(v):
    return str(int(round(v)))


def fmt_ratio(v):
    return f"{v:.2f}"


def fmt_ms(v):
    return fmt_int(v)


def fmt_sec(v):
    return f"{v:.1f}"


def fmt_p(p):
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def fmt_p_star(p):
    s = fmt_p(p)
    if p < 0.001:
        return s + "***"
    if p < 0.01:
        return s + "**"
    if p < 0.05:
        return s + "*"
    return s


DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def safe_macro_name(s):
    """Convert a string like 'congested/reset-at-keyframe/100' to a valid LaTeX macro name.

    LaTeX command names cannot contain digits, so we convert digit sequences
    to word form: 100 -> OneHundred, 200 -> TwoHundred, 500 -> FiveHundred, etc.
    """
    s = s.replace("-", "").replace("/", "").replace("_", "")
    s = s.replace("1000", "OneThousand")
    s = s.replace("100", "OneHundred")
    s = s.replace("200", "TwoHundred")
    s = s.replace("500", "FiveHundred")
    for d, w in DIGIT_WORDS.items():
        s = s.replace(d, w)
    return s



def collect_per_run_summary(rows):
    """Return dict: (cond, strat, tmo) -> list of per-run dicts."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["strategy"], int(r["timeout_ms"]))
        groups[key].append({
            "delivery_ratio": float(r["delivery_ratio"]),
            "recv_total": int(r["recv_total"]),
            "recv_i": int(r["recv_i"]),
            "recv_p": int(r["recv_p"]),
            "objects_dropped": int(r["objects_dropped"]),
            "groups_reset": int(r["groups_reset"]),
        })
    return dict(groups)


def aggregate_abandonment(rows):
    """Return dict: (cond, strat, tmo) -> mean abandonment metrics across runs."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["strategy"], int(r["timeout_ms"]))].append(r)
    out = {}
    for key, rs in groups.items():
        fr = [float(r["recv_frac_reset"]) for r in rs if r["recv_frac_reset"] != ""]
        tr = [float(r["trigger_obj_median"]) for r in rs if r["trigger_obj_median"] != ""]
        out[key] = {
            "reset_groups": np.mean([float(r["reset_groups"]) for r in rs]),
            "complete_groups": np.mean([float(r["complete_groups"]) for r in rs]),
            "kf_in_reset": np.mean([float(r["kf_in_reset"]) for r in rs]),
            "recv_frac_reset": np.mean(fr) if fr else 0.0,
            "trigger_obj_median": np.median(tr) if tr else 0.0,
        }
    return out


def collect_per_run_latency(rows):
    """Return dict: (cond, strat, tmo) -> list of per-run mean I-frame latencies."""
    by_run = defaultdict(list)
    for r in rows:
        if r["frame_type"] != "I":
            continue
        key = (r["condition"], r["strategy"], int(r["timeout_ms"]), int(r["run"]))
        by_run[key].append(float(r["latency_ms"]))

    groups = defaultdict(list)
    for (cond, strat, tmo, run), vals in sorted(by_run.items()):
        config_key = (cond, strat, tmo)
        groups[config_key].append(np.mean(vals))
    return dict(groups)


def collect_per_run_stalls(rows):
    """Return dict: (cond, strat, tmo) -> list of per-run stall dicts."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["strategy"], int(r["timeout_ms"]))
        groups[key].append({
            "stall_count": int(r["stall_count"]),
            "stall_total_s": float(r["stall_total_s"]),
        })
    return dict(groups)


def collect_per_run_per_group_kf_latency(rows):
    """Return dict: (cond, strat, tmo, run) -> {group_id: latency_ms}.

    Only I-frames. Used for matched-group and penalty-imputed comparisons.
    """
    result = defaultdict(dict)
    for r in rows:
        if r["frame_type"] != "I":
            continue
        key = (r["condition"], r["strategy"], int(r["timeout_ms"]), int(r["run"]))
        result[key][int(r["group_id"])] = float(r["latency_ms"])
    return dict(result)



def aggregate_with_stats(per_run_summary, per_run_latency):
    """Compute mean, std, se, ci95 for each configuration."""
    result = {}
    for key, runs in per_run_summary.items():
        n = len(runs)
        dr = np.array([r["delivery_ratio"] for r in runs])
        recv_i = np.array([r["recv_i"] for r in runs])
        recv_p = np.array([r["recv_p"] for r in runs])
        recv_total = np.array([r["recv_total"] for r in runs])
        resets = np.array([r["groups_reset"] for r in runs])
        dropped = np.array([r["objects_dropped"] for r in runs])

        entry = {
            "runs": n,
            "recv_total_mean": np.mean(recv_total),
            "recv_total_std": np.std(recv_total, ddof=1) if n > 1 else 0.0,
            "recv_i_mean": np.mean(recv_i),
            "recv_i_std": np.std(recv_i, ddof=1) if n > 1 else 0.0,
            "recv_p_mean": np.mean(recv_p),
            "delivery_ratio_mean": np.mean(dr),
            "delivery_ratio_std": np.std(dr, ddof=1) if n > 1 else 0.0,
            "delivery_ratio_se": np.std(dr, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            "groups_reset_mean": np.mean(resets),
            "groups_reset_std": np.std(resets, ddof=1) if n > 1 else 0.0,
            "objects_dropped_mean": np.mean(dropped),
            "_dr_runs": dr,
            "_recv_i_runs": recv_i,
        }
        ci_half = 1.96 * entry["delivery_ratio_se"]
        entry["delivery_ratio_ci_lo"] = entry["delivery_ratio_mean"] - ci_half
        entry["delivery_ratio_ci_hi"] = entry["delivery_ratio_mean"] + ci_half

        lat_runs = per_run_latency.get(key)
        if lat_runs:
            lat = np.array(lat_runs)
            entry["lat_mean"] = np.mean(lat)
            entry["lat_std"] = np.std(lat, ddof=1) if len(lat) > 1 else 0.0
            entry["lat_se"] = entry["lat_std"] / np.sqrt(len(lat)) if len(lat) > 1 else 0.0
            entry["_lat_runs"] = lat
        else:
            entry["lat_mean"] = None
            entry["_lat_runs"] = np.array([])

        result[key] = entry
    return result


def aggregate_latency_all(rows):
    """Aggregate all I-frame latencies (not per-run) for percentile computation."""
    groups = defaultdict(list)
    for r in rows:
        if r["frame_type"] != "I":
            continue
        key = (r["condition"], r["strategy"], int(r["timeout_ms"]))
        groups[key].append(float(r["latency_ms"]))

    result = {}
    for key, vals in groups.items():
        vals = np.array(vals)
        result[key] = {
            "count": len(vals),
            "avg": np.mean(vals),
            "p50": np.median(vals),
            "p95": np.percentile(vals, 95),
            "max": np.max(vals),
        }
    return result



def welch_ttest(a, b):
    """Welch's t-test. Returns (t_stat, p_value, effect_size_cohen_d).
    Handles zero-variance cases gracefully."""
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan

    if np.std(a, ddof=1) == 0 and np.std(b, ddof=1) == 0:
        if np.mean(a) == np.mean(b):
            return 0.0, 1.0, 0.0
        else:
            # Means differ but no variance - effectively infinite t
            return np.inf, 0.0, np.inf

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std > 0:
        d = abs(np.mean(a) - np.mean(b)) / pooled_std
    else:
        d = 0.0 if np.mean(a) == np.mean(b) else np.inf
    return t_stat, p_val, d


def tost_welch(a, b, delta):
    """TOST equivalence test using Welch's t-test.
    Tests H0: |mean(a) - mean(b)| >= delta against H1: |diff| < delta.
    Returns (p_tost, ci90_lo, ci90_hi).
    Equivalence demonstrated if p_tost < 0.05."""
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan

    diff = np.mean(a) - np.mean(b)
    va = np.var(a, ddof=1) / len(a)
    vb = np.var(b, ddof=1) / len(b)
    se = np.sqrt(va + vb)

    if se == 0:
        if abs(diff) < delta:
            return 0.0, diff, diff
        else:
            return 1.0, diff, diff

    df = (va + vb)**2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))

    t_upper = (diff - delta) / se
    p_upper = stats.t.cdf(t_upper, df)
    t_lower = (diff + delta) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_tost = max(p_upper, p_lower)

    # 90% CI (corresponds to alpha=0.05 for two one-sided tests)
    t_crit = stats.t.ppf(0.95, df)
    ci_lo = diff - t_crit * se
    ci_hi = diff + t_crit * se

    return p_tost, ci_lo, ci_hi
    return t_stat, p_val, d


def run_statistical_tests(agg):
    """Run pairwise t-tests for key comparisons.

    For each (condition, timeout), compare:
      1. reset-at-keyframe vs reset-stream
      2. reset-at-object vs reset-stream
      3. reset-at-keyframe vs reset-at-object
    On metrics: delivery_ratio, recv_i (keyframes), per-run mean latency.
    """
    conditions = ["lossy", "congested", "severe"]
    timeouts = [100, 200, 500, 1000]
    comparisons = [
        ("reset-at-keyframe", "reset-stream", "RAK vs RS"),
        ("reset-at-object", "reset-stream", "RAO vs RS"),
        ("reset-at-keyframe", "reset-at-object", "RAK vs RAO"),
    ]
    metrics = [
        ("delivery_ratio", "_dr_runs", "Delivery ratio"),
        ("recv_i", "_recv_i_runs", "Keyframes"),
        ("latency", "_lat_runs", "KF latency"),
    ]

    results = []
    for cond in conditions:
        for tmo in timeouts:
            for strat_a, strat_b, comp_label in comparisons:
                key_a = (cond, strat_a, tmo)
                key_b = (cond, strat_b, tmo)
                a_data = agg.get(key_a)
                b_data = agg.get(key_b)
                if not a_data or not b_data:
                    continue

                for metric_key, run_key, metric_label in metrics:
                    arr_a = a_data[run_key]
                    arr_b = b_data[run_key]
                    if len(arr_a) == 0 or len(arr_b) == 0:
                        continue

                    t_stat, p_val, cohen_d = welch_ttest(arr_a, arr_b)
                    results.append({
                        "condition": cond,
                        "timeout": tmo,
                        "comparison": comp_label,
                        "metric": metric_label,
                        "mean_a": np.mean(arr_a),
                        "mean_b": np.mean(arr_b),
                        "diff": np.mean(arr_a) - np.mean(arr_b),
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "cohen_d": cohen_d,
                        "n_a": len(arr_a),
                        "n_b": len(arr_b),
                    })
    return results


def run_latency_bias_analysis(per_group_kf_lat, gop_duration_ms=1000.0):
    """Matched-group and GoP-penalty-imputed latency comparisons.

    For each (condition, timeout, run), compare RESET_AT vs RESET_STREAM:
      - Matched: only groups both strategies delivered (apples-to-apples)
      - GoP penalty: RESET_STREAM missing keyframes get latency = GoP duration,
        reflecting that the viewer freezes for one GoP until the next keyframe

    Returns a list of result dicts for the report.
    """
    conditions = ["lossy", "congested", "severe"]
    timeouts = [100, 200, 500, 1000]
    num_groups = 32
    at_strategies = ["reset-at-keyframe", "reset-at-object"]

    results = []
    for cond in conditions:
        for tmo in timeouts:
            for at_strat in at_strategies:
                at_label = "RAK" if at_strat == "reset-at-keyframe" else "RAO"

                matched_at_runs = []
                matched_rs_runs = []
                penalty_rs_runs = []
                raw_at_runs = []
                raw_rs_runs = []

                for run in range(1, 11):
                    at_key = (cond, at_strat, tmo, run)
                    rs_key = (cond, "reset-stream", tmo, run)
                    at_groups = per_group_kf_lat.get(at_key, {})
                    rs_groups = per_group_kf_lat.get(rs_key, {})

                    if not at_groups:
                        continue

                    if at_groups:
                        raw_at_runs.append(np.mean(list(at_groups.values())))
                    if rs_groups:
                        raw_rs_runs.append(np.mean(list(rs_groups.values())))

                    common = set(at_groups.keys()) & set(rs_groups.keys())
                    if common:
                        matched_at_runs.append(np.mean([at_groups[g] for g in common]))
                        matched_rs_runs.append(np.mean([rs_groups[g] for g in common]))

                    # GoP penalty: missing keyframes cost one GoP duration
                    penalty_vals = []
                    for g in range(num_groups):
                        if g in rs_groups:
                            penalty_vals.append(rs_groups[g])
                        else:
                            penalty_vals.append(gop_duration_ms)
                    penalty_rs_runs.append(np.mean(penalty_vals))

                if not matched_at_runs:
                    continue

                raw_at = np.array(raw_at_runs)
                raw_rs = np.array(raw_rs_runs)
                matched_at = np.array(matched_at_runs)
                matched_rs = np.array(matched_rs_runs)
                penalty_rs = np.array(penalty_rs_runs)

                t_matched, p_matched, d_matched = welch_ttest(matched_at, matched_rs)
                t_penalty, p_penalty, d_penalty = welch_ttest(raw_at, penalty_rs)

                # TOST equivalence test (50ms margin = 5% of GoP penalty)
                tost_p, tost_ci_lo, tost_ci_hi = tost_welch(matched_at, matched_rs, delta=50.0)

                results.append({
                    "condition": cond,
                    "timeout": tmo,
                    "at_strategy": at_label,
                    "raw_at_mean": np.mean(raw_at),
                    "raw_rs_mean": np.mean(raw_rs) if len(raw_rs) > 0 else np.nan,
                    "matched_at_mean": np.mean(matched_at),
                    "matched_rs_mean": np.mean(matched_rs),
                    "matched_diff": np.mean(matched_at) - np.mean(matched_rs),
                    "matched_t": t_matched,
                    "matched_p": p_matched,
                    "matched_d": d_matched,
                    "tost_p": tost_p,
                    "tost_ci_lo": tost_ci_lo,
                    "tost_ci_hi": tost_ci_hi,
                    "penalty_rs_mean": np.mean(penalty_rs),
                    "penalty_diff": np.mean(raw_at) - np.mean(penalty_rs),
                    "penalty_t": t_penalty,
                    "penalty_p": p_penalty,
                    "penalty_d": d_penalty,
                })
    return results



def _kv(key, value):
    return f"\\pgfkeyssetvalue{{/val/{key}}}{{{value}}}"


def generate_values(agg, latency_all, decodability, bias_results, abandonment, outdir):
    """Write values.tex with pgfkeys key-value definitions.

    Keys use the format: condition/strategy/timeout/metric
    Access in LaTeX via \\val{condition/strategy/timeout/metric}
    """
    lines = [
        "% Auto-generated by gen-eval-data.py -- do not edit",
        "% Access values via \\val{condition/strategy/timeout/metric}",
        "% Example: \\val{congested/reset-at-keyframe/100/lat-run-avg}",
    ]

    for (cond, strat, tmo), s in sorted(agg.items()):
        p = f"{cond}/{strat}/{tmo}"
        lines.append(f"% {cond} / {strat} / {tmo}ms")
        lines.append(_kv(f"{p}/recv-total", fmt_int(s['recv_total_mean'])))
        lines.append(_kv(f"{p}/recv-total-std", fmt_ratio(s['recv_total_std'])))
        lines.append(_kv(f"{p}/recv-i", f"{s['recv_i_mean']:.1f}"))
        lines.append(_kv(f"{p}/recv-i-std", fmt_ratio(s['recv_i_std'])))
        lines.append(_kv(f"{p}/recv-p", fmt_int(s['recv_p_mean'])))
        lines.append(_kv(f"{p}/ratio", fmt_ratio(s['delivery_ratio_mean'])))
        lines.append(_kv(f"{p}/ratio-std", fmt_ratio(s['delivery_ratio_std'])))
        lines.append(_kv(f"{p}/ratio-se", fmt_ratio(s['delivery_ratio_se'])))
        lines.append(_kv(f"{p}/ratio-ci-lo", fmt_ratio(s['delivery_ratio_ci_lo'])))
        lines.append(_kv(f"{p}/ratio-ci-hi", fmt_ratio(s['delivery_ratio_ci_hi'])))
        lines.append(_kv(f"{p}/resets", fmt_int(s['groups_reset_mean'])))
        lines.append(_kv(f"{p}/resets-std", fmt_ratio(s['groups_reset_std'])))
        lines.append(_kv(f"{p}/dropped", fmt_int(s['objects_dropped_mean'])))

        lat = latency_all.get((cond, strat, tmo))
        if lat:
            lines.append(_kv(f"{p}/lat-avg", fmt_ms(lat['avg'])))
            lines.append(_kv(f"{p}/lat-median", fmt_ms(lat['p50'])))
            lines.append(_kv(f"{p}/lat-p95", fmt_ms(lat['p95'])))
            lines.append(_kv(f"{p}/lat-max", fmt_ms(lat['max'])))
            lines.append(_kv(f"{p}/lat-count", str(lat['count'])))
        if s["lat_mean"] is not None:
            lines.append(_kv(f"{p}/lat-run-avg", fmt_ms(s['lat_mean'])))
            lines.append(_kv(f"{p}/lat-run-std", fmt_ms(s['lat_std'])))
            lines.append(_kv(f"{p}/lat-run-se", fmt_ms(s['lat_se'])))

    lines.append("% === Decodability ===")
    for r in decodability:
        tmo = int(r["timeout_ms"])
        p = f"{r['condition']}/{r['strategy']}/{tmo}"
        lines.append(_kv(f"{p}/decod-sec", fmt_sec(float(r['avg_decodable_s']))))
        lines.append(_kv(f"{p}/decod-ratio", fmt_ratio(float(r['avg_decodable_ratio']))))
        lines.append(_kv(f"{p}/stall-count", fmt_sec(float(r['avg_stall_count']))))
        lines.append(_kv(f"{p}/stall-total-sec", fmt_sec(float(r['avg_stall_total_s']))))
        lines.append(_kv(f"{p}/stall-max-sec", fmt_sec(float(r['avg_stall_max_s']))))

    lines.append("% === Bias analysis (matched & GoP-penalty) ===")
    for r in bias_results:
        at_strat = "reset-at-keyframe" if r["at_strategy"] == "RAK" else "reset-at-object"
        label = "rak" if r["at_strategy"] == "RAK" else "rao"
        p = f"{r['condition']}/{at_strat}/{r['timeout']}"
        rsp = f"{r['condition']}/reset-stream/{r['timeout']}"

        lines.append(f"% {r['condition']} / {label} vs rs / {r['timeout']}ms")
        lines.append(_kv(f"{p}/matched-lat", fmt_ms(r['matched_at_mean'])))
        lines.append(_kv(f"{rsp}/matched-lat/{label}", fmt_ms(r['matched_rs_mean'])))
        lines.append(_kv(f"{p}/matched-p/{label}", fmt_p(r['matched_p'])))
        lines.append(_kv(f"{p}/tost-p/{label}", fmt_p(r['tost_p'])))
        lines.append(_kv(f"{p}/ci-lo/{label}", fmt_ms(r['tost_ci_lo'])))
        lines.append(_kv(f"{p}/ci-hi/{label}", fmt_ms(r['tost_ci_hi'])))
        lines.append(_kv(f"{rsp}/penalty-lat/{label}", fmt_ms(r['penalty_rs_mean'])))
        lines.append(_kv(f"{p}/penalty-p/{label}", fmt_p(r['penalty_p'])))

    lines.append("% === Abandonment (reset groups) ===")
    for (cond, strat, tmo), a in sorted(abandonment.items()):
        p = f"{cond}/{strat}/{tmo}"
        lines.append(_kv(f"{p}/complete-groups", fmt_int(a['complete_groups'])))
        lines.append(_kv(f"{p}/reset-kf-delivered", fmt_int(a['kf_in_reset'])))
        lines.append(_kv(f"{p}/reset-recv-frac", fmt_int(a['recv_frac_reset'] * 100)))
        lines.append(_kv(f"{p}/reset-trigger-obj", fmt_int(a['trigger_obj_median'])))

    with open(outdir / "values.tex", "w") as f:
        f.write("\n".join(lines) + "\n")



def strategy_label(s):
    mapping = {
        "none": "None",
        "reset-stream": r"\resetstream",
        "reset-at-keyframe": r"\resetat KF",
        "reset-at-object": r"\resetat obj",
    }
    return mapping.get(s, s)


def condition_label(c):
    labels = {"congested": "Cong."}
    return labels.get(c, c.capitalize())


def generate_results_table(agg, latency_all, outdir):
    """Write tab_results.tex with mean +/- std."""
    conditions = ["lossy", "congested", "severe"]
    strategies = ["none", "reset-at-keyframe", "reset-at-object", "reset-stream"]

    lines = [
        "% Auto-generated by gen-eval-data.py - do not edit",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evaluation results (mean $\pm$ std over 10 runs). KF\,=\,keyframes received (of 32). Latency values are for keyframes.}",
        r"\label{tab:results}",
        r"\begin{tabular}{llr rrl rrrr}",
        r"\hline",
        r"\textbf{Condition} & \textbf{Strategy} & \textbf{TMO} & \textbf{Recv} & \textbf{KF} & \textbf{Ratio} & \textbf{Avg} & \textbf{P50} & \textbf{P95} & \textbf{Max} \\",
        r" & & \textbf{(ms)} & & & & \multicolumn{4}{c}{\textbf{Keyframe latency (ms)}} \\",
        r"\hline",
    ]

    for ci, cond in enumerate(conditions):
        first_in_cond = True
        for strat in strategies:
            tmo_list = [0] if strat == "none" else [100, 200, 500, 1000]
            for tmo in tmo_list:
                key = (cond, strat, tmo)
                s = agg.get(key)
                lat = latency_all.get(key)
                if not s:
                    continue

                cond_col = condition_label(cond) if first_in_cond else ""
                first_in_cond = False
                strat_col = strategy_label(strat)
                tmo_col = "--" if tmo == 0 else str(tmo)

                dr_str = fmt_ratio(s["delivery_ratio_mean"])
                if s["delivery_ratio_std"] > 0.005:
                    dr_str = f"${dr_str} \\pm {fmt_ratio(s['delivery_ratio_std'])}$"

                lat_avg = fmt_ms(lat["avg"]) if lat else "--"
                lat_p50 = fmt_ms(lat["p50"]) if lat else "--"
                lat_p95 = fmt_ms(lat["p95"]) if lat else "--"
                lat_max = fmt_ms(lat["max"]) if lat else "--"

                kf_str = fmt_int(s["recv_i_mean"])
                if s["recv_i_std"] > 0.5:
                    kf_str = f"${kf_str} \\pm {fmt_int(s['recv_i_std'])}$"

                lines.append(
                    f"{cond_col} & {strat_col} & {tmo_col} & "
                    f"{fmt_int(s['recv_total_mean'])} & {kf_str} & "
                    f"{dr_str} & "
                    f"{lat_avg} & {lat_p50} & {lat_p95} & {lat_max} \\\\"
                )
        if ci < len(conditions) - 1:
            lines.append(r"\hline")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    with open(outdir / "tab_results.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_stalls_table(decodability, outdir):
    """Write tab_stalls.tex - playback continuity table."""
    conditions = ["lossy", "congested", "severe"]
    strategies = ["none", "reset-at-keyframe", "reset-at-object", "reset-stream"]

    decodability_idx = {}
    for r in decodability:
        key = (r["condition"], r["strategy"], int(r["timeout_ms"]))
        decodability_idx[key] = r

    lines = [
        "% Auto-generated by gen-eval-data.py - do not edit",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Playback continuity: decodable video duration and stall behavior, averaged over 10 runs.}",
        r"\label{tab:stalls}",
        r"\begin{tabular}{llr rrr rr}",
        r"\hline",
        r"\textbf{Condition} & \textbf{Strategy} & \textbf{TMO} & \textbf{Decodable} & \textbf{Decod.} & \textbf{Stalls} & \textbf{Stall} & \textbf{Max stall} \\",
        r" & & \textbf{(ms)} & \textbf{(s)} & \textbf{ratio} & & \textbf{total (s)} & \textbf{(s)} \\",
        r"\hline",
    ]

    for ci, cond in enumerate(conditions):
        first_in_cond = True
        for strat in strategies:
            tmo_list = [0] if strat == "none" else [100, 200, 500, 1000]
            for tmo in tmo_list:
                key = (cond, strat, tmo)
                r = decodability_idx.get(key)
                if not r:
                    continue

                cond_col = condition_label(cond) if first_in_cond else ""
                first_in_cond = False
                strat_col = strategy_label(strat)
                tmo_col = "--" if tmo == 0 else str(tmo)

                lines.append(
                    f"{cond_col} & {strat_col} & {tmo_col} & "
                    f"{fmt_sec(float(r['avg_decodable_s']))} & "
                    f"{fmt_ratio(float(r['avg_decodable_ratio']))} & "
                    f"{fmt_sec(float(r['avg_stall_count']))} & "
                    f"{fmt_sec(float(r['avg_stall_total_s']))} & "
                    f"{fmt_sec(float(r['avg_stall_max_s']))} \\\\"
                )
        if ci < len(conditions) - 1:
            lines.append(r"\hline")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    with open(outdir / "tab_decodability.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_compact_table(agg, latency_all, decodability, outdir):
    """Write tab_compact.tex - compact table showing selected timeouts."""
    conditions = ["lossy", "congested", "severe"]
    strategies = ["none", "reset-at-keyframe", "reset-at-object", "reset-stream"]
    show_timeouts = [100, 1000]

    decodability_idx = {}
    for r in decodability:
        key = (r["condition"], r["strategy"], int(r["timeout_ms"]))
        decodability_idx[key] = r

    lines = [
        "% Auto-generated by gen-eval-data.py - do not edit",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Selected results: delivery, latency, and playback impact at \SI{100}{\milli\second} and \SI{1000}{\milli\second} timeouts, averaged over 10 runs.}",
        r"\label{tab:compact}",
        r"\begin{tabular}{llr rr rr rr}",
        r"\hline",
        r"\textbf{Condition} & \textbf{Strategy} & \textbf{TMO} & \textbf{Recv} & \textbf{KF} & \textbf{Avg lat.} & \textbf{P95 lat.} & \textbf{Decod.} & \textbf{Stalls} \\",
        r" & & \textbf{(ms)} & & \textbf{/32} & \multicolumn{2}{c}{\textbf{(ms)}} & \textbf{(s)} & \\",
        r"\hline",
    ]

    for ci, cond in enumerate(conditions):
        first_in_cond = True
        for strat in strategies:
            tmo_list = [0] if strat == "none" else show_timeouts
            for tmo in tmo_list:
                key = (cond, strat, tmo)
                s = agg.get(key)
                lat = latency_all.get(key)
                d = decodability_idx.get(key)
                if not s:
                    continue

                cond_col = condition_label(cond) if first_in_cond else ""
                first_in_cond = False
                strat_col = strategy_label(strat)
                tmo_col = "--" if tmo == 0 else str(tmo)

                lat_avg = fmt_ms(lat["avg"]) if lat else "--"
                lat_p95 = fmt_ms(lat["p95"]) if lat else "--"
                decod = fmt_sec(float(d["avg_decodable_s"])) if d else "--"
                stalls = fmt_sec(float(d["avg_stall_count"])) if d else "--"

                lines.append(
                    f"{cond_col} & {strat_col} & {tmo_col} & "
                    f"{fmt_int(s['recv_total_mean'])} & {fmt_int(s['recv_i_mean'])} & "
                    f"{lat_avg} & {lat_p95} & "
                    f"{decod} & {stalls} \\\\"
                )
        if ci < len(conditions) - 1:
            lines.append(r"\hline")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    with open(outdir / "tab_compact.tex", "w") as f:
        f.write("\n".join(lines) + "\n")



def generate_stats_table(test_results, outdir):
    """Write tab_stats.tex - p-value table for key comparisons."""
    lines = [
        "% Auto-generated by gen-eval-data.py - do not edit",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Welch's $t$-test results for pairwise strategy comparisons (10 runs each). $d$\,=\,Cohen's $d$ effect size. "
        r"Significance: {*}\,$p<0.05$, {**}\,$p<0.01$, {***}\,$p<0.001$.}",
        r"\label{tab:stats}",
        r"\begin{tabular}{llllrrrl}",
        r"\hline",
        r"\textbf{Condition} & \textbf{TMO (ms)} & \textbf{Comparison} & \textbf{Metric} & \textbf{Mean A} & \textbf{Mean B} & \textbf{$d$} & \textbf{$p$} \\",
        r"\hline",
    ]

    prev_cond = None
    for r in test_results:
        if r["condition"] == "baseline":
            continue
        if np.isnan(r["p_value"]):
            continue

        cond_col = condition_label(r["condition"]) if r["condition"] != prev_cond else ""
        prev_cond = r["condition"]

        if r["metric"] == "Delivery ratio":
            mean_a = fmt_ratio(r["mean_a"])
            mean_b = fmt_ratio(r["mean_b"])
        elif r["metric"] == "Keyframes":
            mean_a = fmt_int(r["mean_a"])
            mean_b = fmt_int(r["mean_b"])
        else:
            mean_a = fmt_ms(r["mean_a"])
            mean_b = fmt_ms(r["mean_b"])

        d_str = f"{r['cohen_d']:.2f}" if not np.isinf(r["cohen_d"]) else r"$\infty$"
        p_str = fmt_p_star(r["p_value"])

        lines.append(
            f"{cond_col} & {r['timeout']} & {r['comparison']} & {r['metric']} & "
            f"{mean_a} & {mean_b} & {d_str} & {p_str} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    with open(outdir / "tab_stats.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_bias_table(bias_results, outdir):
    """Write tab_bias.tex - survivorship bias correction for KF latency."""
    # Only show RAK vs RS (RAO vs RS is nearly identical)
    rows = [r for r in bias_results if r["at_strategy"] == "RAK"]

    lines = [
        "% Auto-generated by gen-eval-data.py - do not edit",
        r"\begin{tabular}{ll rrrr rrr}",
        r"\hline",
        r" & & "
        r"\multicolumn{4}{c}{\textbf{Matched (ms)}} & "
        r"\multicolumn{3}{c}{\textbf{GoP penalty (ms)}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-9}",
        r"\textbf{Cond.} & \textbf{TMO} & "
        r"\textbf{RAK} & \textbf{RS} & \textbf{90\% CI} & \textbf{$p_{\,\text{TOST}}$} & "
        r"\textbf{RAK} & \textbf{RS} & \textbf{$p$} \\",
        r"\hline",
    ]

    prev_cond = None
    for r in rows:
        cond_col = condition_label(r["condition"]) if r["condition"] != prev_cond else ""
        prev_cond = r["condition"]

        ci_str = f"[{r['tost_ci_lo']:+.0f}, {r['tost_ci_hi']:+.0f}]"

        lines.append(
            f"{cond_col} & {r['timeout']} & "
            f"{fmt_ms(r['matched_at_mean'])} & {fmt_ms(r['matched_rs_mean'])} & "
            f"{ci_str} & {fmt_p(r['tost_p'])} & "
            f"{fmt_ms(r['raw_at_mean'])} & {fmt_ms(r['penalty_rs_mean'])} & "
            f"{fmt_p(r['penalty_p'])} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}",
    ]

    with open(outdir / "tab_bias.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_stats_report(test_results, agg, bias_results, outdir):
    """Write stats_report.txt - human-readable statistical summary."""
    lines = [
        "Statistical Analysis Report",
        "=" * 60,
        "",
        "Welch's t-test (unequal variance), two-tailed",
        "Significance levels: * p<0.05, ** p<0.01, *** p<0.001",
        "Effect size: Cohen's d (small=0.2, medium=0.5, large=0.8)",
        "",
    ]

    lines.append("=" * 60)
    lines.append("VARIANCE SUMMARY (delivery ratio, std dev over 10 runs)")
    lines.append("=" * 60)
    for key in sorted(agg.keys()):
        cond, strat, tmo = key
        s = agg[key]
        if cond == "baseline":
            continue
        lines.append(
            f"  {cond:12s} {strat:22s} {tmo:5d}ms  "
            f"mean={s['delivery_ratio_mean']:.4f}  "
            f"std={s['delivery_ratio_std']:.4f}  "
            f"se={s['delivery_ratio_se']:.4f}  "
            f"95%CI=[{s['delivery_ratio_ci_lo']:.4f}, {s['delivery_ratio_ci_hi']:.4f}]"
        )
    lines.append("")

    lines.append("=" * 60)
    lines.append("PAIRWISE COMPARISONS")
    lines.append("=" * 60)
    prev_group = None
    for r in test_results:
        if r["condition"] == "baseline":
            continue
        group = (r["condition"], r["timeout"])
        if group != prev_group:
            lines.append("")
            lines.append(f"--- {r['condition']} / {r['timeout']}ms ---")
            prev_group = group

        sig = ""
        if not np.isnan(r["p_value"]):
            if r["p_value"] < 0.001:
                sig = " ***"
            elif r["p_value"] < 0.01:
                sig = " **"
            elif r["p_value"] < 0.05:
                sig = " *"
            else:
                sig = " (ns)"

        d_str = f"{r['cohen_d']:.2f}" if not np.isinf(r["cohen_d"]) else "inf"
        p_str = f"{r['p_value']:.6f}" if not np.isnan(r["p_value"]) else "N/A"

        lines.append(
            f"  {r['comparison']:12s}  {r['metric']:16s}  "
            f"A={r['mean_a']:.4f}  B={r['mean_b']:.4f}  "
            f"diff={r['diff']:+.4f}  "
            f"t={r['t_stat']:+.3f}  p={p_str}  d={d_str}{sig}"
        )

    lines.append("")
    lines.append("=" * 60)
    lines.append("KEY FINDINGS")
    lines.append("=" * 60)

    sig_kf = [r for r in test_results
              if r["comparison"] == "RAK vs RS"
              and r["metric"] == "Keyframes"
              and not np.isnan(r["p_value"])
              and r["p_value"] < 0.05]
    lines.append(f"\nRESET_AT KF vs RESET_STREAM on keyframe delivery:")
    lines.append(f"  Significant in {len(sig_kf)} out of "
                 f"{len([r for r in test_results if r['comparison'] == 'RAK vs RS' and r['metric'] == 'Keyframes' and r['condition'] != 'baseline'])} "
                 f"non-baseline comparisons")

    sig_rak_rao = [r for r in test_results
                   if r["comparison"] == "RAK vs RAO"
                   and not np.isnan(r["p_value"])
                   and r["p_value"] < 0.05]
    lines.append(f"\nRESET_AT KF vs RESET_AT obj:")
    lines.append(f"  Significant in {len(sig_rak_rao)} out of "
                 f"{len([r for r in test_results if r['comparison'] == 'RAK vs RAO' and r['condition'] != 'baseline'])} "
                 f"non-baseline comparisons")
    for r in sig_rak_rao:
        lines.append(f"    {r['condition']}/{r['timeout']}ms {r['metric']}: p={r['p_value']:.6f}, d={r['cohen_d']:.2f}")

    zero_var = [(k, v) for k, v in agg.items()
                if k[0] == "congested" and k[1] != "none"
                and v["delivery_ratio_std"] < 0.001]
    if zero_var:
        lines.append(f"\nNOTE: {len(zero_var)} congested configurations have zero variance in delivery ratio")
        lines.append("  (deterministic bandwidth constraint → identical results across runs)")
        lines.append("  t-tests for these yield p=1.0 when comparing strategies with identical outcomes,")
        lines.append("  or p=0.0 when means differ with zero variance.")

    lines.append("")
    lines.append("=" * 60)
    lines.append("LATENCY SURVIVORSHIP BIAS ANALYSIS")
    lines.append("=" * 60)
    lines.append("")
    lines.append("RESET_STREAM reports lower avg KF latency because it only delivers")
    lines.append("the 'easy' (low-latency) keyframes. Two corrections:")
    lines.append("  Matched:     compare only groups both strategies delivered")
    lines.append("  GoP penalty: assign missing keyframes latency = 1 GoP duration (~1s)")
    lines.append("               (viewer freezes until next GoP's keyframe)")
    lines.append("")

    for r in bias_results:
        if r["at_strategy"] != "RAK":
            continue  # Only show RAK vs RS to avoid duplication

        sig_m = ""
        if not np.isnan(r["matched_p"]):
            if r["matched_p"] < 0.001: sig_m = "***"
            elif r["matched_p"] < 0.01: sig_m = "**"
            elif r["matched_p"] < 0.05: sig_m = "*"
            else: sig_m = "(ns)"

        sig_p = ""
        if not np.isnan(r["penalty_p"]):
            if r["penalty_p"] < 0.001: sig_p = "***"
            elif r["penalty_p"] < 0.01: sig_p = "**"
            elif r["penalty_p"] < 0.05: sig_p = "*"
            else: sig_p = "(ns)"

        lines.append(f"--- {r['condition']} / {r['timeout']}ms ---")
        lines.append(f"  Raw:         RAK={r['raw_at_mean']:.1f}ms  RS={r['raw_rs_mean']:.1f}ms  (RS appears {r['raw_at_mean'] - r['raw_rs_mean']:+.1f}ms faster)")
        lines.append(f"  Matched:     RAK={r['matched_at_mean']:.1f}ms  RS={r['matched_rs_mean']:.1f}ms  diff={r['matched_diff']:+.1f}ms  p={fmt_p(r['matched_p'])} {sig_m}")
        lines.append(f"  GoP penalty: RAK={r['raw_at_mean']:.1f}ms  RS(pen)={r['penalty_rs_mean']:.1f}ms  diff={r['penalty_diff']:+.1f}ms  p={fmt_p(r['penalty_p'])} {sig_p}")
        lines.append("")

    with open(outdir / "stats_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")



def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results-dir> [<output-dir>]", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("gen")
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(results_dir / "summary.csv")
    latency_rows = read_csv(results_dir / "latency.csv")
    decodability_rows = read_csv(results_dir / "decodability.csv")
    abandonment = aggregate_abandonment(read_csv(results_dir / "abandonment.csv"))

    per_run_summary = collect_per_run_summary(summary_rows)
    per_run_latency = collect_per_run_latency(latency_rows)
    per_group_kf_lat = collect_per_run_per_group_kf_latency(latency_rows)
    agg = aggregate_with_stats(per_run_summary, per_run_latency)
    latency_all = aggregate_latency_all(latency_rows)

    test_results = run_statistical_tests(agg)
    bias_results = run_latency_bias_analysis(per_group_kf_lat)

    generate_values(agg, latency_all, decodability_rows, bias_results, abandonment, outdir)
    generate_results_table(agg, latency_all, outdir)
    generate_stalls_table(decodability_rows, outdir)
    generate_compact_table(agg, latency_all, decodability_rows, outdir)
    generate_stats_table(test_results, outdir)
    generate_bias_table(bias_results, outdir)
    generate_stats_report(test_results, agg, bias_results, outdir)

    print(f"Generated files in {outdir}/:")
    for f in sorted(outdir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
