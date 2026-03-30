#!/usr/bin/env python3
"""Gate long AdaEvolve runs on hard mini-suite quality thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _collect(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    grouped: Dict[Tuple[str, str], Dict[str, float]] = {}
    counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        seed = str(row.get("seed_profile", "unknown"))
        result = row.get("result") or {}
        if not isinstance(result, dict):
            continue
        key = (model, seed)
        bucket = grouped.setdefault(
            key,
            {
                "completion_rate": 0.0,
                "mean_task_score": 0.0,
                "n_scaffold_crash": 0.0,
                "verification_rate": 0.0,
            },
        )
        counts[key] = counts.get(key, 0) + 1
        for metric in ("completion_rate", "mean_task_score", "n_scaffold_crash", "verification_rate"):
            bucket[metric] += float(result.get(metric, 0.0))
    for key, bucket in grouped.items():
        n = max(1, counts.get(key, 1))
        for metric in tuple(bucket.keys()):
            bucket[metric] = bucket[metric] / n
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate hard mini-suite readiness")
    parser.add_argument("summary_json", help="Output of run_hard_mini_suite.py")
    parser.add_argument("--min-completion-rate", type=float, default=0.20)
    parser.add_argument("--min-mean-score", type=float, default=0.20)
    parser.add_argument("--max-crashes", type=float, default=0.5)
    parser.add_argument("--min-verification-rate", type=float, default=0.7)
    args = parser.parse_args()

    rows = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    grouped = _collect(rows)
    if not grouped:
        print("FAIL: no evaluable rows found.")
        return 2

    passed = False
    for (model, seed), metrics in sorted(grouped.items()):
        ok = (
            metrics["completion_rate"] >= args.min_completion_rate
            and metrics["mean_task_score"] >= args.min_mean_score
            and metrics["n_scaffold_crash"] <= args.max_crashes
            and metrics["verification_rate"] >= args.min_verification_rate
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} model={model} seed={seed} "
            f"completion={metrics['completion_rate']:.3f} "
            f"mean_score={metrics['mean_task_score']:.3f} "
            f"crashes={metrics['n_scaffold_crash']:.3f} "
            f"verification={metrics['verification_rate']:.3f}"
        )
        passed = passed or ok

    if passed:
        print("At least one seed/model passed the gate; long evolution can proceed.")
        return 0

    print("No seed/model passed the gate; continue seed hardening before long evolution.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
