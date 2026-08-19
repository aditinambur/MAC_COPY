#!/usr/bin/env python
"""
Aggregate results_log JSONL rows from onpolicy/scripts/phase2_3_repair.py into a per-config
summary table (n, acceptance rate, mean/std reward & comm recovery), and optionally run a
Welch's t-test between two named configs.

This does NOT run any experiments -- it only reads what --results_log has already written.
Run phase2_3_repair.py with --results_log <path> (the same path, or several different paths
across a sweep) first, then point this script at those file(s).

Example:
  python onpolicy/scripts/aggregate_repair_results.py results/causal.jsonl results/reward_only.jsonl \
    --compare "causal|target=auto|threads=32|mirror=partner_full" \
              "reward_only|target=comm|threads=32|mirror=partner_full" \
    --csv_out results/summary.csv

Group labels are printed by this script itself (one per distinct config found in the input);
copy them verbatim into --compare.
"""

import argparse
import csv
import glob
import json
import statistics
import sys


def _load_rows(paths):
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rows.append(json.loads(line))
            except FileNotFoundError:
                print("WARNING: no such file: {}".format(path), file=sys.stderr)
    return rows


def _extract_outcome(row):
    """
    Normalize the three final_status shapes phase2_3_repair.py can produce into a common
    (accepted, reward_recovery, comm_recovery, target_used) tuple.
      - manual / reward_only: final_status has reward_recovery/comm_recovery/target directly.
      - causal: final_status has escalation_log; the LAST attempt is the one that determined
        the outcome (either the accepted one, or the final rejection after the ladder was
        exhausted).
      - no_repair_control / not_degraded: no repair ran, no recovery numbers exist.

    A recovery value may also be null: accept_repair reports None for any metric that did not
    lose ground under degradation, since "fraction of the loss recovered" is undefined when
    there was no loss. Such rows are dropped here rather than coerced to a number -- averaging
    them in would invent a recovery that was never measured.
    """
    fs = row.get("final_status") or {}
    mode = fs.get("mode")

    if mode in ("manual", "reward_only"):
        if "reward_recovery" not in fs:
            return None
        outcome = (bool(fs.get("accepted")), fs["reward_recovery"], fs["comm_recovery"],
                   fs.get("target"))
    elif mode == "causal":
        log = fs.get("escalation_log") or []
        if not log:
            return None
        last = log[-1]
        outcome = (bool(fs.get("accepted")), last["reward_recovery"], last["comm_recovery"],
                   fs.get("accepted_target") or last.get("target"))
    else:
        return None  # no_repair_control / not_degraded / unknown: nothing to aggregate

    if outcome[1] is None or outcome[2] is None:
        return None
    return outcome


def _group_key(row):
    target = row.get("repair_target_arg")
    return "{}|target={}|threads={}|mirror={}".format(
        row.get("controller"),
        target if target is not None else "auto",
        row.get("n_rollout_threads"),
        row.get("mirror_scope"),
    )


def _mean_std(values):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)


def _summarize(rows):
    groups = {}
    for row in rows:
        outcome = _extract_outcome(row)
        if outcome is None:
            continue
        key = _group_key(row)
        groups.setdefault(key, []).append(outcome)

    summary = {}
    for key, outcomes in groups.items():
        accepted_flags = [o[0] for o in outcomes]
        reward_rec = [o[1] for o in outcomes]
        comm_rec = [o[2] for o in outcomes]
        r_mean, r_std = _mean_std(reward_rec)
        c_mean, c_std = _mean_std(comm_rec)
        summary[key] = {
            "n": len(outcomes),
            "accept_rate": sum(accepted_flags) / len(accepted_flags),
            "reward_recovery_mean": r_mean, "reward_recovery_std": r_std,
            "comm_recovery_mean": c_mean, "comm_recovery_std": c_std,
            "reward_recovery_values": reward_rec,
            "comm_recovery_values": comm_rec,
        }
    return summary


def _print_table(summary):
    print("\n{:<55} {:>4} {:>10} {:>14} {:>14}".format(
        "config", "n", "accept%", "reward_rec", "comm_rec"))
    print("-" * 100)
    for key in sorted(summary):
        s = summary[key]
        print("{:<55} {:>4} {:>9.0f}% {:>6.1f}%±{:>4.1f} {:>6.1f}%±{:>4.1f}".format(
            key[:55], s["n"], 100 * s["accept_rate"],
            100 * s["reward_recovery_mean"], 100 * (s["reward_recovery_std"] or 0),
            100 * s["comm_recovery_mean"], 100 * (s["comm_recovery_std"] or 0)))
    if not summary:
        print("(no rows with a completed repair attempt found -- check --results_log paths)")


def _write_csv(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n", "accept_rate", "reward_recovery_mean",
                         "reward_recovery_std", "comm_recovery_mean", "comm_recovery_std"])
        for key in sorted(summary):
            s = summary[key]
            writer.writerow([key, s["n"], s["accept_rate"], s["reward_recovery_mean"],
                             s["reward_recovery_std"], s["comm_recovery_mean"],
                             s["comm_recovery_std"]])
    print("\nWrote {}".format(path))


def _compare(summary, label_a, label_b):
    if label_a not in summary or label_b not in summary:
        missing = [l for l in (label_a, label_b) if l not in summary]
        print("\nERROR: --compare label(s) not found in the aggregated data: {}".format(missing))
        print("Copy a label verbatim from the 'config' column above.")
        return

    try:
        from scipy import stats
    except ImportError:
        print("\n(scipy not installed -- printing descriptive comparison only, no t-test.)")
        stats = None

    a, b = summary[label_a], summary[label_b]
    print("\n=== Comparison ===")
    print("A: {}".format(label_a))
    print("B: {}".format(label_b))
    for metric in ("reward_recovery", "comm_recovery"):
        va, vb = a["{}_values".format(metric)], b["{}_values".format(metric)]
        print("\n{}:  A mean={:.1%} (n={})   B mean={:.1%} (n={})".format(
            metric, statistics.mean(va), len(va), statistics.mean(vb), len(vb)))
        if stats is not None and len(va) >= 2 and len(vb) >= 2:
            t, p = stats.ttest_ind(va, vb, equal_var=False)  # Welch's t-test
            print("  Welch's t-test: t={:.3f}  p={:.4f}{}".format(
                t, p, "  (p<0.05)" if p < 0.05 else ""))
        else:
            print("  (need >=2 samples per group for a t-test; have {} and {})".format(
                len(va), len(vb)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_log", nargs="+",
                        help="one or more --results_log JSONL paths (globs allowed).")
    parser.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"), default=None,
                        help="run a Welch's t-test between two config labels printed by "
                             "this script (requires scipy; falls back to descriptive-only).")
    parser.add_argument("--csv_out", type=str, default=None,
                        help="also write the summary table to this CSV path.")
    args = parser.parse_args()

    rows = _load_rows(args.results_log)
    print("Loaded {} result row(s) from {} file pattern(s).".format(len(rows), len(args.results_log)))

    summary = _summarize(rows)
    _print_table(summary)

    if args.csv_out:
        _write_csv(summary, args.csv_out)

    if args.compare:
        _compare(summary, args.compare[0], args.compare[1])


if __name__ == "__main__":
    main()
