#!/usr/bin/env python3
"""Run a curated hard-task mini-suite for fast seed iteration."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = REPO_ROOT / "benchmarks" / "multiagent" / "terminal_bench" / "evaluator.py"
DEFAULT_PROGRAM = REPO_ROOT / "benchmarks" / "multiagent" / "terminal_bench" / "initial_program.py"


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text).strip("-_") or "default"


def _default_output_path(split_profile: str, models: List[str], seed_profiles: List[str], timeout: int) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_part = _slug("-".join(models[:2]))
    seed_part = _slug("-".join(seed_profiles[:2]))
    filename = f"{_slug(split_profile)}_{model_part}_{seed_part}_t{timeout}_{timestamp}.json"
    return REPO_ROOT / "outputs" / "terminal_bench_adaevolve" / filename


def run_eval(program: Path, split: str, env: Dict[str, str]) -> Dict[str, object]:
    cmd = ["uv", "run", "python", str(EVALUATOR), "--split", split, str(program)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    out: Dict[str, object] = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
    }
    if proc.stdout.strip():
        try:
            out["result"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            out["stdout"] = proc.stdout.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TB2 hard mini-suite")
    parser.add_argument("--program", default=str(DEFAULT_PROGRAM))
    parser.add_argument("--task-root", default=os.getenv("TBENCH_TASK_ROOT", "/home/mertcemri/tb2_harbor_wrapped"))
    parser.add_argument("--models", nargs="+", default=[os.getenv("SOLVER_MODEL", "gpt-5")])
    parser.add_argument("--seed-profiles", nargs="+", default=["single_agent_native", "two_phase"])
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    parser.add_argument("--timeout", type=int, default=420, help="Per-task timeout in seconds")
    parser.add_argument(
        "--split-profile",
        default=os.getenv("TBENCH_SPLIT_PROFILE", "tb2_hardmini"),
        help="Split manifest profile prefix",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Where to write the final JSON summary. Defaults to a unique timestamped path.",
    )
    parser.add_argument(
        "--progress-jsonl",
        default="",
        help="Optional JSONL path for incremental per-run progress. Defaults next to the final summary.",
    )
    args = parser.parse_args()

    program = Path(args.program).resolve()
    output = Path(args.output).resolve() if args.output else _default_output_path(
        split_profile=args.split_profile,
        models=args.models,
        seed_profiles=args.seed_profiles,
        timeout=args.timeout,
    )
    progress_jsonl = (
        Path(args.progress_jsonl).resolve()
        if args.progress_jsonl
        else output.with_suffix(".jsonl")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if progress_jsonl.exists():
        progress_jsonl.unlink()

    rows: List[Dict[str, object]] = []
    for model in args.models:
        for seed_profile in args.seed_profiles:
            env = os.environ.copy()
            env.update(
                {
                    "SOLVER_MODEL": model,
                    "TBENCH_TASK_ROOT": args.task_root,
                    "TBENCH_SPLIT_PROFILE": args.split_profile,
                    "TBENCH_SPLIT_STRATEGY": "balanced",
                    "TBENCH_SEED_PROFILE": seed_profile,
                    "TRAIN_TASKS": "9999",
                    "DEV_TASKS": "9999",
                    "TEST_TASKS": "9999",
                    "CANDIDATE_TIMEOUT_S": str(args.timeout),
                }
            )
            for split in args.splits:
                print(f"[hard-mini] model={model} seed={seed_profile} split={split}", flush=True)
                row = {
                    "model": model,
                    "seed_profile": seed_profile,
                    "split": split,
                    **run_eval(program, split, env),
                }
                rows.append(row)
                with progress_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row))
                    fh.write("\n")

    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote hard mini-suite summary to {output}")
    print(f"Wrote incremental progress to {progress_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
