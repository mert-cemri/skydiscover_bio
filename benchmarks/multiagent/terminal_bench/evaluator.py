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
from collections import defaultdict
import hashlib
import json
import os
import random
import re
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
_SPLIT_ROOT = _HERE / "splits"
_TASK_CACHE: Dict[str, List[Dict[str, Any]]] | None = None
_EVALUATOR_CACHE: Dict[tuple[str, int], HarborEvaluator] = {}
_EVALUATOR_LOCKS: Dict[tuple[str, int], threading.Lock] = {}
_CACHE_LOCK = threading.Lock()
MAX_STEPS_CAP = 30
EFFICIENCY_BONUS_WEIGHT = float(os.getenv("TBENCH_EFFICIENCY_BONUS", "0.02"))
VERIFICATION_BONUS_WEIGHT = float(os.getenv("TBENCH_VERIFICATION_BONUS", "0.03"))
AGENT_PENALTY_PER_EXTRA = float(os.getenv("TBENCH_AGENT_PENALTY", "0.01"))


def _is_harbor_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "instruction.md").exists()
        and (path / "tests" / "test.sh").exists()
        and (path / "environment" / "Dockerfile").exists()
    )


def _parse_task_metadata(task_dir: Path) -> Dict[str, Any]:
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return {}
    try:
        text = toml_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    def _extract(pattern: str) -> str:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    difficulty = _extract(r'difficulty\s*=\s*"([^"]+)"')
    category = _extract(r'category\s*=\s*"([^"]+)"')
    timeout_text = _extract(r"\[agent\][\s\S]*?timeout_sec\s*=\s*([0-9.]+)")
    tags_match = re.search(r"tags\s*=\s*\[([^\]]*)\]", text)
    tags = re.findall(r'"([^"]+)"', tags_match.group(1)) if tags_match else []
    metadata: Dict[str, Any] = {}
    if difficulty:
        metadata["difficulty"] = difficulty
    if category:
        metadata["category"] = category
    if tags:
        metadata["tags"] = tags
    if timeout_text:
        try:
            metadata["agent_timeout_sec"] = float(timeout_text)
        except ValueError:
            pass
    return metadata


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
                    "metadata": _parse_task_metadata(current),
                }
            )
            dirnames[:] = []
    tasks.sort(key=lambda item: item["id"])
    return tasks


def _split_tasks_balanced(tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Create train/dev/test splits that preserve category+difficulty diversity."""
    seed = int(os.getenv("TBENCH_SPLIT_SEED", "42"))
    train_ratio = float(os.getenv("TBENCH_TRAIN_RATIO", "0.7"))
    dev_ratio = float(os.getenv("TBENCH_DEV_RATIO", "0.15"))
    rng = random.Random(seed)
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        meta = task.get("metadata", {})
        key = (meta.get("category", "unknown"), meta.get("difficulty", "unknown"))
        buckets[key].append(task)

    train: List[Dict[str, Any]] = []
    dev: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda item: item["id"])
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = max(1, int(round(n * train_ratio))) if n > 1 else n
        remaining = max(0, n - n_train)
        n_dev = min(remaining, int(round(n * dev_ratio)))
        if remaining > 0 and n_dev == 0 and n >= 5:
            n_dev = 1
        n_dev = min(n_dev, max(0, n - n_train - 1)) if n >= 3 else n_dev
        train.extend(ordered[:n_train])
        dev.extend(ordered[n_train : n_train + n_dev])
        test.extend(ordered[n_train + n_dev :])

    train.sort(key=lambda item: item["id"])
    dev.sort(key=lambda item: item["id"])
    test.sort(key=lambda item: item["id"])
    return {"train": train, "dev": dev, "test": test}


def _load_profile_split(root: Path, profile: str) -> Dict[str, List[Dict[str, Any]]]:
    manifests = {
        split: _SPLIT_ROOT / f"{profile}_{split}.txt" for split in ("train", "dev", "test")
    }
    if not any(path.exists() for path in manifests.values()):
        return {}
    task_map = {task["id"]: task for task in _discover_harbor_tasks(root)}
    split_tasks: Dict[str, List[Dict[str, Any]]] = {}
    for split, manifest_path in manifests.items():
        ids: List[str] = []
        if manifest_path.exists():
            ids = [
                line.strip()
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        split_tasks[split] = [task_map[task_id] for task_id in ids if task_id in task_map]
    return split_tasks


def _split_external_tasks(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    train_dir = root / "train"
    test_dir = root / "test"
    split_strategy = os.getenv("TBENCH_SPLIT_STRATEGY", "hash").strip().lower()
    split_profile = os.getenv("TBENCH_SPLIT_PROFILE", "").strip()
    if train_dir.exists() or test_dir.exists():
        split = {
            "train": _discover_harbor_tasks(train_dir),
            "dev": [],
            "test": _discover_harbor_tasks(test_dir),
        }
        if split_strategy == "balanced":
            merged = split["train"] + split["test"]
            return _split_tasks_balanced(merged)
        return split

    all_tasks = _discover_harbor_tasks(root)
    if split_profile:
        profiled = _load_profile_split(root, split_profile)
        if profiled:
            return profiled
    if split_strategy == "balanced":
        return _split_tasks_balanced(all_tasks)
    ratio = float(os.getenv("TBENCH_TRAIN_RATIO", "0.6"))
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for task in all_tasks:
        task_hash = int(hashlib.md5(task["id"].encode("utf-8")).hexdigest(), 16)
        if (task_hash % 1000) / 1000.0 < ratio:
            train.append(task)
        else:
            test.append(task)
    return {"train": train, "dev": [], "test": test}


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
            "dev": [],
            "test": _discover_harbor_tasks(_TASK_ROOT / "test"),
        }
    return _TASK_CACHE


def _get_tasks(split: str) -> List[Dict[str, Any]]:
    return list(_load_task_pool().get(split, []))


def _make_harbor_evaluator(task_dir: str, timeout_s: int) -> HarborEvaluator:
    cfg = EvaluatorConfig(
        evaluation_file=task_dir,
        file_suffix=".py",
        timeout=timeout_s,
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


def _resolve_task_timeout(task_def: Dict[str, Any]) -> int:
    default_timeout = int(os.getenv("CANDIDATE_TIMEOUT_S", "240"))
    max_timeout_raw = os.getenv("TBENCH_MAX_TASK_TIMEOUT", "").strip()
    max_timeout = int(max_timeout_raw) if max_timeout_raw.isdigit() else 0
    meta = task_def.get("metadata", {}) or {}
    meta_timeout = meta.get("agent_timeout_sec")
    resolved = default_timeout
    if meta_timeout is not None:
        try:
            resolved = max(default_timeout, int(float(meta_timeout)))
        except (TypeError, ValueError):
            resolved = default_timeout
    if max_timeout > 0:
        resolved = min(resolved, max_timeout)
    return resolved


def _get_harbor_evaluator(task_dir: str, timeout_s: int) -> HarborEvaluator:
    key = (task_dir, timeout_s)
    with _CACHE_LOCK:
        evaluator = _EVALUATOR_CACHE.get(key)
        if evaluator is None:
            evaluator = _make_harbor_evaluator(task_dir, timeout_s)
            _EVALUATOR_CACHE[key] = evaluator
            _EVALUATOR_LOCKS[key] = threading.Lock()
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

    timeout_s = _resolve_task_timeout(task_def)
    evaluator = _get_harbor_evaluator(task_def["task_dir"], timeout_s)
    lock = _EVALUATOR_LOCKS[(task_def["task_dir"], timeout_s)]
    with lock:
        result = asyncio.run(
            evaluator.evaluate_program(program_source, program_id=task_def["id"], mode=mode)
        )

    metrics = dict(result.metrics)
    score = float(metrics.get("combined_score", 0.0))
    scaffold_exit_code = int(float(metrics.get("scaffold_exit_code", 0.0)))
    completed = score >= 0.999 and scaffold_exit_code == 0
    return {
        "task_id": task_def["id"],
        "completed": completed,
        "score": score,
        "metrics": metrics,
        "artifact_summary": _condense_artifacts(result),
        "metadata": dict(task_def.get("metadata", {})),
    }


def _build_feedback(split: str, per_task: List[Dict[str, Any]]) -> str:
    lines = [f"Terminal bench {split} evaluation summary:"]
    for item in per_task[:8]:
        status = "PASS" if item["completed"] else "FAIL"
        meta = item.get("metadata", {})
        meta_bits = []
        if meta.get("category"):
            meta_bits.append(f"category={meta['category']}")
        if meta.get("difficulty"):
            meta_bits.append(f"difficulty={meta['difficulty']}")
        lines.append(
            f"- {item['task_id']}: {status}, score={item['score']:.3f}, "
            f"steps={int(item['metrics'].get('total_steps', 0))}, "
            f"done_signal={int(item['metrics'].get('done_signal', 0.0))}, "
            f"agents={int(item['metrics'].get('num_agents', 0))}"
            + (f", {', '.join(meta_bits)}" if meta_bits else "")
        )
        if item["artifact_summary"]:
            lines.append(item["artifact_summary"][:800])
    return "\n".join(lines)


def _classify_failure(task_result: Dict[str, Any]) -> str:
    if task_result.get("completed"):
        return "completed"
    metrics = task_result.get("metrics", {}) or {}
    score = float(task_result.get("score", 0.0))
    artifact = str(task_result.get("artifact_summary", ""))
    if int(float(metrics.get("scaffold_exit_code", 0.0))) != 0:
        if "BadRequestError" in artifact:
            return "tool_error"
        return "scaffold_crash"
    if bool(metrics.get("timeout", False)):
        return "timeout"
    if "Unknown tool" in artifact or "tool" in artifact.lower() and "error" in artifact.lower():
        return "tool_error"
    if float(metrics.get("done_signal", 0.0)) >= 1.0 and score < 0.999:
        return "verifier_miss"
    if float(metrics.get("total_steps", 0.0)) <= 0:
        return "no_progress"
    if score > 0.0:
        return "partial"
    return "task_failure"


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
    n_verified_completions = 0
    n_scaffold_crash = 0
    failure_taxonomy: Dict[str, int] = defaultdict(int)

    try:
        for task_def in tasks:
            task_result = _run_task(program_path, task_def, mode=mode)
            failure_reason = _classify_failure(task_result)
            task_result["failure_reason"] = failure_reason
            per_task.append(task_result)
            total_score += task_result["score"]
            metrics = task_result["metrics"]
            total_steps += float(metrics.get("total_steps", 0.0))
            total_latency += float(metrics.get("latency_s", 0.0))
            num_agents = max(num_agents, int(metrics.get("num_agents", 0)))
            has_done_signal = float(metrics.get("done_signal", 0.0)) >= 1.0
            if task_result["completed"]:
                n_completed += 1
                if has_done_signal:
                    n_verified_completions += 1
            if has_done_signal:
                n_done_signal += 1
            if int(float(metrics.get("scaffold_exit_code", 0.0))) != 0:
                n_scaffold_crash += 1
            failure_taxonomy[failure_reason] += 1
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
    verification_rate = float(n_verified_completions / n_completed) if n_completed else 0.0
    efficiency_bonus = (
        max(0.0, (1.0 - avg_steps / MAX_STEPS_CAP) * EFFICIENCY_BONUS_WEIGHT) if n_completed else 0.0
    )
    verification_bonus = verification_rate * VERIFICATION_BONUS_WEIGHT if n_completed else 0.0
    agent_penalty = max(0, num_agents - 6) * AGENT_PENALTY_PER_EXTRA
    combined_score = max(0.0, mean_reward + efficiency_bonus + verification_bonus - agent_penalty)

    metrics = {
        "combined_score": float(combined_score),
        "completion_rate": float(n_completed / max(1, n_attempted)),
        "mean_task_score": float(mean_reward),
        "n_completed": float(n_completed),
        "n_attempted": float(n_attempted),
        "n_done_signal": float(n_done_signal),
        "n_verified_completions": float(n_verified_completions),
        "verification_rate": float(verification_rate),
        "n_scaffold_crash": float(n_scaffold_crash),
        "n_timeout": float(failure_taxonomy.get("timeout", 0)),
        "n_tool_error": float(failure_taxonomy.get("tool_error", 0)),
        "n_verifier_miss": float(failure_taxonomy.get("verifier_miss", 0)),
        "n_no_progress": float(failure_taxonomy.get("no_progress", 0)),
        "n_task_failure": float(failure_taxonomy.get("task_failure", 0)),
        "n_partial": float(failure_taxonomy.get("partial", 0)),
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
        "failure_taxonomy": json.dumps(dict(sorted(failure_taxonomy.items())), indent=2),
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
    task_score = float(probe.metrics.get("mean_task_score", 0.0))
    stage_ok = (
        task_score > 0.0
        and float(probe.metrics.get("n_scaffold_crash", 0.0)) == 0.0
        and "Evaluation crashed:" not in probe.artifacts.get("feedback", "")
    )
    # Use actual task score so stage1 filters candidates that run but solve nothing.
    # A scaffold that crashes gets 0.0; one that runs but fails gets 0.0 (filtered at
    # 0.05 cascade threshold); one that solves the task gets up to 0.1.
    score = task_score * 0.1 if stage_ok else 0.0
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
    elif split == "dev":
        raw = os.getenv("DEV_TASKS", "").strip()
        if not raw:
            return tasks
        num_tasks = int(raw)
        random.seed(77)
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


def evaluate_dev(program_path: str) -> EvaluationResult:
    return _evaluate_split(program_path, "dev")


def evaluate(program_path: str) -> EvaluationResult:
    return _evaluate_split(program_path, "train")


def evaluate_test(program_path: str) -> EvaluationResult:
    return _evaluate_split(program_path, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harbor-backed terminal bench evaluator")
    parser.add_argument("program_path", help="Path to the candidate program")
    parser.add_argument("--test", action="store_true", help="Run held-out test tasks")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        help="Run an explicit split instead of the default train/test shortcut.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks in the selected split instead of sampling",
    )
    args = parser.parse_args()

    if args.all:
        selected_split = args.split or ("test" if args.test else "train")
        if selected_split == "test":
            os.environ.pop("TEST_TASKS", None)
        elif selected_split == "dev":
            os.environ["TEST_TASKS"] = str(len(_get_tasks("dev")))
        else:
            os.environ["TRAIN_TASKS"] = str(len(_get_tasks("train")))

    selected_split = args.split or ("test" if args.test else "train")
    if selected_split == "test":
        result = evaluate_test(args.program_path)
    elif selected_split == "dev":
        result = evaluate_dev(args.program_path)
    else:
        result = evaluate(args.program_path)
    print(json.dumps(result.to_dict(), indent=2))
