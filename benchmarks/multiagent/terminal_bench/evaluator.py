"""
Harbor-backed evaluator for the terminal scaffold benchmark.

This benchmark now evaluates candidates on Harbor-format Docker tasks rather
than host-local shell scripts. It ships with a bundled mini-suite under
`tasks/train` and `tasks/test`, and can optionally use an external task root.

Environment variables:
  SOLVER_MODEL          model for the evolved scaffold (default: gpt-4o-mini)
  TRAIN_TASKS           number of train tasks per evaluation (default: 3)
  TEST_TASKS            number of test tasks for final evaluation (default: all)
  CANDIDATE_TIMEOUT_S   timeout per Harbor task (default: 240)
  TBENCH_TASK_ROOT      optional external Harbor task root
  TBENCH_TRAIN_RATIO    hash split when external root lacks train/test dirs
"""

import argparse
import atexit
import asyncio
import hashlib
import json
import os
import random
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.evaluation.harbor_evaluator import HarborEvaluator

_HERE = Path(__file__).resolve().parent
_TASK_ROOT = _HERE / "tasks"
_TASK_CACHE: Dict[str, List[Dict[str, Any]]] | None = None
_EVALUATOR_CACHE: Dict[str, HarborEvaluator] = {}
_EVALUATOR_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()
MAX_STEPS_CAP = 30


def _is_harbor_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "instruction.md").exists()
        and (path / "tests" / "test.sh").exists()
        and (path / "environment" / "Dockerfile").exists()
    )


def _discover_harbor_tasks(root: Path) -> List[Dict[str, Any]]:
    if not root.exists():
        return []

    tasks: List[Dict[str, Any]] = []
    for current_root, dirnames, _ in os.walk(root):
        current = Path(current_root)
        if _is_harbor_task_dir(current):
            task_id = str(current.relative_to(root))
            tasks.append(
                {
                    "id": task_id.replace(os.sep, "__"),
                    "task_dir": str(current),
                    "instruction_path": str(current / "instruction.md"),
                }
            )
            dirnames[:] = []
    tasks.sort(key=lambda item: item["id"])
    return tasks


def _split_external_tasks(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    train_dir = root / "train"
    test_dir = root / "test"
    if train_dir.exists() or test_dir.exists():
        return {
            "train": _discover_harbor_tasks(train_dir),
            "test": _discover_harbor_tasks(test_dir),
        }

    all_tasks = _discover_harbor_tasks(root)
    ratio = float(os.getenv("TBENCH_TRAIN_RATIO", "0.6"))
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for task in all_tasks:
        task_hash = int(hashlib.md5(task["id"].encode("utf-8")).hexdigest(), 16)
        if (task_hash % 1000) / 1000.0 < ratio:
            train.append(task)
        else:
            test.append(task)
    return {"train": train, "test": test}


def _load_task_pool() -> Dict[str, List[Dict[str, Any]]]:
    global _TASK_CACHE
    if _TASK_CACHE is not None:
        return _TASK_CACHE

    external_root = os.getenv("TBENCH_TASK_ROOT", "").strip()
    if external_root:
        root = Path(external_root).expanduser().resolve()
        _TASK_CACHE = _split_external_tasks(root)
    else:
        _TASK_CACHE = {
            "train": _discover_harbor_tasks(_TASK_ROOT / "train"),
            "test": _discover_harbor_tasks(_TASK_ROOT / "test"),
        }
    return _TASK_CACHE


def _get_tasks(split: str) -> List[Dict[str, Any]]:
    return list(_load_task_pool().get(split, []))


def _make_harbor_evaluator(task_dir: str) -> HarborEvaluator:
    cfg = EvaluatorConfig(
        evaluation_file=task_dir,
        file_suffix=".py",
        timeout=int(os.getenv("CANDIDATE_TIMEOUT_S", "240")),
        max_retries=0,
        cascade_evaluation=False,
    )
    return HarborEvaluator(task_dir, cfg, max_concurrent=1)


def _close_evaluators() -> None:
    for evaluator in list(_EVALUATOR_CACHE.values()):
        try:
            evaluator.close()
        except Exception:
            pass


atexit.register(_close_evaluators)


def _get_harbor_evaluator(task_dir: str) -> HarborEvaluator:
    with _CACHE_LOCK:
        evaluator = _EVALUATOR_CACHE.get(task_dir)
        if evaluator is None:
            evaluator = _make_harbor_evaluator(task_dir)
            _EVALUATOR_CACHE[task_dir] = evaluator
            _EVALUATOR_LOCKS[task_dir] = threading.Lock()
        return evaluator


def _condense_artifacts(result: EvaluationResult) -> str:
    chunks: List[str] = []
    for key in ("error", "stderr", "stdout", "test_stderr", "test_stdout"):
        value = result.artifacts.get(key)
        if value:
            text = str(value).strip()
            if text:
                chunks.append(f"{key}: {text[:1200]}")
    return "\n".join(chunks)


def _run_task(program_path: str, task_def: Dict[str, Any], mode: str) -> Dict[str, Any]:
    with open(program_path, "r", encoding="utf-8") as f:
        program_source = f.read()

    evaluator = _get_harbor_evaluator(task_def["task_dir"])
    lock = _EVALUATOR_LOCKS[task_def["task_dir"]]
    with lock:
        result = asyncio.run(
            evaluator.evaluate_program(program_source, program_id=task_def["id"], mode=mode)
        )

    metrics = dict(result.metrics)
    score = float(metrics.get("combined_score", 0.0))
    completed = score >= 0.999
    return {
        "task_id": task_def["id"],
        "completed": completed,
        "score": score,
        "metrics": metrics,
        "artifact_summary": _condense_artifacts(result),
    }


def _build_feedback(split: str, per_task: List[Dict[str, Any]]) -> str:
    lines = [f"Terminal bench {split} evaluation summary:"]
    for item in per_task[:8]:
        status = "PASS" if item["completed"] else "FAIL"
        lines.append(
            f"- {item['task_id']}: {status}, score={item['score']:.3f}, "
            f"steps={int(item['metrics'].get('total_steps', 0))}, "
            f"done_signal={int(item['metrics'].get('done_signal', 0.0))}, "
            f"agents={int(item['metrics'].get('num_agents', 0))}"
        )
        if item["artifact_summary"]:
            lines.append(item["artifact_summary"][:800])
    return "\n".join(lines)


def _evaluate_on_tasks(program_path: str, tasks: List[Dict[str, Any]], mode: str) -> EvaluationResult:
    start = time.time()
    if not os.path.exists(program_path):
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"feedback": f"Program file not found: {program_path}"},
        )

    per_task: List[Dict[str, Any]] = []
    total_score = 0.0
    total_steps = 0.0
    total_latency = 0.0
    num_agents = 0
    n_completed = 0
    n_done_signal = 0

    try:
        for task_def in tasks:
            task_result = _run_task(program_path, task_def, mode=mode)
            per_task.append(task_result)
            total_score += task_result["score"]
            metrics = task_result["metrics"]
            total_steps += float(metrics.get("total_steps", 0.0))
            total_latency += float(metrics.get("latency_s", 0.0))
            num_agents = max(num_agents, int(metrics.get("num_agents", 0)))
            if task_result["completed"]:
                n_completed += 1
            if float(metrics.get("done_signal", 0.0)) >= 1.0:
                n_done_signal += 1
    except Exception as exc:
        traceback.print_exc()
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "eval_wall_s": float(time.time() - start),
            },
            artifacts={"feedback": f"Evaluation crashed: {exc}"},
        )

    n_attempted = len(tasks)
    mean_reward = total_score / max(1, n_attempted)
    avg_steps = total_steps / max(1, n_attempted)
    efficiency_bonus = max(0.0, (1.0 - avg_steps / MAX_STEPS_CAP) * 0.05) if n_completed else 0.0
    agent_penalty = max(0, num_agents - 6) * 0.01
    combined_score = max(0.0, mean_reward + efficiency_bonus - agent_penalty)

    metrics = {
        "combined_score": float(combined_score),
        "completion_rate": float(n_completed / max(1, n_attempted)),
        "mean_task_score": float(mean_reward),
        "n_completed": float(n_completed),
        "n_attempted": float(n_attempted),
        "n_done_signal": float(n_done_signal),
        "num_agents": float(num_agents),
        "total_steps": float(total_steps),
        "avg_steps_per_task": float(avg_steps),
        "total_latency_s": float(total_latency),
        "avg_latency_s": float(total_latency / max(1, n_attempted)),
        "eval_wall_s": float(time.time() - start),
    }
    artifacts = {
        "feedback": _build_feedback(mode, per_task),
        "per_task": json.dumps(per_task, indent=2),
        "task_ids": json.dumps([task["id"] for task in tasks]),
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_stage1(program_path: str) -> EvaluationResult:
    tasks = _get_tasks("train")
    if not tasks:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"feedback": "No train Harbor tasks available."},
        )
    probe = _evaluate_on_tasks(program_path, [_choose_stage1_task(tasks)], mode="train")
    stage_ok = (
        float(probe.metrics.get("n_attempted", 0.0)) >= 1.0
        and "Evaluation crashed:" not in probe.artifacts.get("feedback", "")
    )
    score = 0.1 if stage_ok else 0.0
    return EvaluationResult(
        metrics={
            "combined_score": score,
            "stage1_ok": 1.0 if stage_ok else 0.0,
            "stage1_task_score": float(probe.metrics.get("mean_task_score", 0.0)),
            "eval_wall_s": float(probe.metrics.get("eval_wall_s", 0.0)),
        },
        artifacts=probe.artifacts,
    )


def _sample_tasks(split: str) -> List[Dict[str, Any]]:
    tasks = _get_tasks(split)
    if not tasks:
        return []
    if split == "train":
        num_tasks = int(os.getenv("TRAIN_TASKS", "3"))
        random.seed(42)
    else:
        raw = os.getenv("TEST_TASKS", "").strip()
        if not raw:
            return tasks
        num_tasks = int(raw)
        random.seed(99)
    return random.sample(tasks, min(num_tasks, len(tasks)))


def _choose_stage1_task(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    preferred = ("mkdir", "json", "git")
    return min(
        tasks,
        key=lambda task: (
            next(
                (idx for idx, token in enumerate(preferred) if token in task["id"]),
                len(preferred),
            ),
            len(task["id"]),
        ),
    )


def _evaluate_split(program_path: str, split: str) -> EvaluationResult:
    tasks = _sample_tasks(split)
    if not tasks:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"feedback": f"No {split} Harbor tasks available."},
        )
    mode = "train" if split == "train" else "test"
    result = _evaluate_on_tasks(program_path, tasks, mode=mode)
    result.artifacts.setdefault("split", split)
    return result


def evaluate_stage2(program_path: str) -> EvaluationResult:
    return _evaluate_split(program_path, "train")


def evaluate(program_path: str) -> EvaluationResult:
    return _evaluate_split(program_path, "train")


def evaluate_test(program_path: str) -> EvaluationResult:
    return _evaluate_split(program_path, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harbor-backed terminal bench evaluator")
    parser.add_argument("program_path", help="Path to the candidate program")
    parser.add_argument("--test", action="store_true", help="Run held-out test tasks")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks in the selected split instead of sampling",
    )
    args = parser.parse_args()

    if args.all:
        if args.test:
            os.environ.pop("TEST_TASKS", None)
        else:
            os.environ["TRAIN_TASKS"] = str(len(_get_tasks("train")))

    result = evaluate_test(args.program_path) if args.test else evaluate(args.program_path)
    print(json.dumps(result.to_dict(), indent=2))
