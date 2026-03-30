#!/usr/bin/env python3
"""Run a small ablation matrix for terminal_bench scaffold variants.

This script is designed for controlled, evaluator-level ablations before
spending large AdaEvolve budgets. It compares seed profiles and optionally
multiple models on train/dev/test TB2-aligned splits.
"""

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
DEFAULT_PROGRAM = REPO_ROOT / "benchmarks" / "multiagent" / "terminal_bench" / "initial_program.py"
DEFAULT_EVALUATOR = REPO_ROOT / "benchmarks" / "multiagent" / "terminal_bench" / "evaluator.py"
DEFAULT_SAMPLE_ROOT = REPO_ROOT / "benchmarks" / "multiagent" / "terminal_bench" / "tasksets" / "terminal-bench-sample-2.0"


def _is_harbor_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "instruction.md").exists()
        and (path / "tests" / "test.sh").exists()
        and (path / "environment" / "Dockerfile").exists()
    )


def _build_env_overrides(model: str, seed_profile: str, task_root: str, split_profile: str) -> Dict[str, str]:
    env_overrides: Dict[str, str] = {
        "SOLVER_MODEL": model,
        "TBENCH_TASK_ROOT": task_root,
    }
    task_root_path = Path(task_root).expanduser().resolve()
    if split_profile:
        # Profile manifests are keyed for multi-task roots. If task_root points to a
        # single Harbor task, disable profile loading so ablation runs evaluate it.
        if not _is_harbor_task_dir(task_root_path):
            env_overrides["TBENCH_SPLIT_PROFILE"] = split_profile
        else:
            env_overrides["TBENCH_SPLIT_STRATEGY"] = "balanced"
    if seed_profile:
        env_overrides["TBENCH_SEED_PROFILE"] = seed_profile
    return env_overrides


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text).strip("-_") or "default"


def _default_output_path(split_profile: str, models: List[str], seed_profiles: List[str]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_part = _slug("-".join(models[:2]))
    seed_part = _slug("-".join(seed_profiles[:2]))
    filename = f"ablation_{_slug(split_profile)}_{model_part}_{seed_part}_{timestamp}.json"
    return REPO_ROOT / "outputs" / "terminal_bench_adaevolve" / filename


def run_eval(program_path: Path, split: str, env_overrides: Dict[str, str]) -> Dict:
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [
        "uv",
        "run",
        "python",
        str(DEFAULT_EVALUATOR),
        "--split",
        split,
        str(program_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    result: Dict[str, object] = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
    }
    if proc.stdout.strip():
        try:
            result["result"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            result["stdout"] = proc.stdout.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluator-level terminal_bench ablations")
    parser.add_argument(
        "--program",
        default=str(DEFAULT_PROGRAM),
        help="Candidate program to evaluate",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Splits to evaluate",
    )
    parser.add_argument(
        "--seed-profiles",
        nargs="+",
        default=["single_agent_native", "two_phase"],
        help="TBENCH_SEED_PROFILE values to test",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[os.getenv("SOLVER_MODEL", "gpt-5.2")],
        help="SOLVER_MODEL values to test",
    )
    parser.add_argument(
        "--task-root",
        default=os.getenv("TBENCH_TASK_ROOT", str(DEFAULT_SAMPLE_ROOT)),
        help="Task root for evaluation",
    )
    parser.add_argument(
        "--split-profile",
        default=os.getenv("TBENCH_SPLIT_PROFILE", "tb2sample"),
        help="Split profile manifest prefix",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Where to write the summary JSON. Defaults to a unique timestamped path.",
    )
    args = parser.parse_args()

    program_path = Path(args.program).resolve()
    matrix: List[Dict[str, object]] = []
    for model in args.models:
        for seed_profile in args.seed_profiles:
            env_overrides = _build_env_overrides(
                model=model,
                seed_profile=seed_profile,
                task_root=args.task_root,
                split_profile=args.split_profile,
            )
            for split in args.splits:
                print(f"[ablation] model={model} seed={seed_profile} split={split}", flush=True)
                eval_result = run_eval(program_path, split, env_overrides)
                matrix.append(
                    {
                        "model": model,
                        "seed_profile": seed_profile,
                        "split": split,
                        "env": env_overrides,
                        **eval_result,
                    }
                )

    output_path = Path(args.output).resolve() if args.output else _default_output_path(
        split_profile=args.split_profile,
        models=args.models,
        seed_profiles=args.seed_profiles,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"Wrote ablation summary to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
