"""Tests for the Harbor-backed terminal bench benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from skydiscover.evaluation.evaluation_result import EvaluationResult


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "benchmarks" / "multiagent" / "terminal_bench"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


terminal_eval = _load_module("terminal_bench_eval_test", BENCH_ROOT / "evaluator.py")
terminal_seed = _load_module("terminal_bench_seed_test", BENCH_ROOT / "initial_program.py")
sample_fetch = _load_module("terminal_bench_sample_fetch_test", BENCH_ROOT / "fetch_terminal_bench_sample.py")


class TestBundledHarborTasks:
    def test_discovers_bundled_train_and_test_tasks(self):
        with patch.object(terminal_eval, "_TASK_CACHE", None):
            train_ids = [task["id"] for task in terminal_eval._get_tasks("train")]
            test_ids = [task["id"] for task in terminal_eval._get_tasks("test")]

        assert len(train_ids) >= 30
        assert len(test_ids) >= 10
        for expected in ("create-directory-tree", "json-transform", "write-bash-script"):
            assert expected in train_ids
        for expected in ("csv-to-json", "python-calculator", "word-count-stats"):
            assert expected in test_ids

    def test_choose_stage1_prefers_simple_probe_task(self):
        tasks = [
            {"id": "git-init-commit"},
            {"id": "json-transform"},
            {"id": "write-bash-script"},
        ]
        chosen = terminal_eval._choose_stage1_task(tasks)
        assert chosen["id"] == "json-transform"

    def test_balanced_split_preserves_tasks(self):
        tasks = [
            {"id": "a1", "metadata": {"category": "git", "difficulty": "easy"}},
            {"id": "a2", "metadata": {"category": "git", "difficulty": "easy"}},
            {"id": "b1", "metadata": {"category": "ml", "difficulty": "hard"}},
            {"id": "b2", "metadata": {"category": "ml", "difficulty": "hard"}},
            {"id": "c1", "metadata": {"category": "systems", "difficulty": "medium"}},
        ]
        split = terminal_eval._split_tasks_balanced(tasks)
        seen = [task["id"] for group in split.values() for task in group]

        assert sorted(seen) == sorted(task["id"] for task in tasks)
        assert len(split["train"]) >= 3

    def test_resolve_task_timeout_uses_metadata_floor(self):
        with patch.dict("os.environ", {"CANDIDATE_TIMEOUT_S": "120"}, clear=False):
            assert terminal_eval._resolve_task_timeout({"metadata": {"agent_timeout_sec": 300}}) == 300
            assert terminal_eval._resolve_task_timeout({"metadata": {"agent_timeout_sec": 60}}) == 120
            assert terminal_eval._resolve_task_timeout({"metadata": {"agent_timeout_sec": "not-a-number"}}) == 120
        with patch.dict(
            "os.environ",
            {"CANDIDATE_TIMEOUT_S": "120", "TBENCH_MAX_TASK_TIMEOUT": "180"},
            clear=False,
        ):
            assert terminal_eval._resolve_task_timeout({"metadata": {"agent_timeout_sec": 300}}) == 180

    def test_tb2sample_manifests_are_disjoint_and_complete(self):
        split_root = BENCH_ROOT / "splits"
        sample_splits = {}
        for split in ("train", "dev", "test"):
            sample_splits[split] = {
                line.strip()
                for line in (split_root / f"tb2sample_{split}.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        combined = set().union(*sample_splits.values())
        assert len(combined) == 10
        assert sum(len(items) for items in sample_splits.values()) == 10
        assert sample_splits["train"].isdisjoint(sample_splits["dev"])
        assert sample_splits["train"].isdisjoint(sample_splits["test"])
        assert sample_splits["dev"].isdisjoint(sample_splits["test"])

    def test_tb2sample_profile_resolves_against_external_root(self, tmp_path):
        all_ids = {
            "build-cython-ext",
            "chess-best-move",
            "configure-git-webserver",
            "fix-code-vulnerability",
            "log-summary-date-ranges",
            "polyglot-c-py",
            "qemu-alpine-ssh",
            "qemu-startup",
            "regex-log",
            "sqlite-with-gcov",
        }
        for task_id in all_ids:
            task_dir = tmp_path / task_id
            (task_dir / "tests").mkdir(parents=True)
            (task_dir / "environment").mkdir(parents=True)
            (task_dir / "instruction.md").write_text("task\n", encoding="utf-8")
            (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
            (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")

        split = terminal_eval._load_profile_split(tmp_path, "tb2sample")
        resolved = {task["id"] for group in split.values() for task in group}
        assert resolved == all_ids
        assert len(split["train"]) == 6
        assert len(split["dev"]) == 2
        assert len(split["test"]) == 2

    def test_fetch_sample_materialize_rejects_existing_output_without_force(self, tmp_path):
        out = tmp_path / "sample"
        out.mkdir()
        try:
            sample_fetch.materialize(out, force=False)
        except RuntimeError as exc:
            assert "Output already exists" in str(exc)
        else:
            raise AssertionError("Expected existing output to raise")


class TestEvaluatorAggregation:
    def test_evaluate_on_tasks_aggregates_metrics_and_feedback(self):
        fake_results = iter(
            [
                {
                    "task_id": "a",
                    "completed": True,
                    "score": 1.0,
                    "metrics": {
                        "combined_score": 1.0,
                        "total_steps": 10.0,
                        "latency_s": 1.5,
                        "num_agents": 4.0,
                        "done_signal": 1.0,
                    },
                    "artifact_summary": "stdout: pass",
                },
                {
                    "task_id": "b",
                    "completed": False,
                    "score": 0.0,
                    "metrics": {
                        "combined_score": 0.0,
                        "total_steps": 20.0,
                        "latency_s": 2.5,
                        "num_agents": 7.0,
                        "done_signal": 0.0,
                    },
                    "artifact_summary": "stderr: fail",
                },
            ]
        )

        with patch.object(terminal_eval, "_run_task", side_effect=lambda *args, **kwargs: next(fake_results)):
            result = terminal_eval._evaluate_on_tasks(
                program_path=__file__,
                tasks=[{"id": "a"}, {"id": "b"}],
                mode="train",
            )

        assert result.metrics["completion_rate"] == 0.5
        assert result.metrics["mean_task_score"] == 0.5
        assert result.metrics["avg_steps_per_task"] == 15.0
        assert result.metrics["num_agents"] == 7.0
        assert result.metrics["n_done_signal"] == 1.0
        assert result.metrics["n_verified_completions"] == 1.0
        assert result.metrics["verification_rate"] == 1.0
        assert result.metrics["n_task_failure"] == 1.0
        assert result.metrics["combined_score"] == 0.53
        assert "Terminal bench train evaluation summary:" in result.artifacts["feedback"]
        assert "task_failure" in result.artifacts["failure_taxonomy"]
        per_task = json.loads(result.artifacts["per_task"])
        assert per_task[0]["task_id"] == "a"
        assert per_task[1]["artifact_summary"] == "stderr: fail"
        assert per_task[1]["failure_reason"] == "task_failure"

    def test_verification_rate_ignores_done_signal_for_failed_tasks(self):
        fake_results = iter(
            [
                {
                    "task_id": "a",
                    "completed": True,
                    "score": 1.0,
                    "metrics": {
                        "combined_score": 1.0,
                        "total_steps": 8.0,
                        "latency_s": 1.2,
                        "num_agents": 1.0,
                        "done_signal": 0.0,
                    },
                    "artifact_summary": "",
                },
                {
                    "task_id": "b",
                    "completed": False,
                    "score": 0.0,
                    "metrics": {
                        "combined_score": 0.0,
                        "total_steps": 8.0,
                        "latency_s": 1.2,
                        "num_agents": 1.0,
                        "done_signal": 1.0,
                    },
                    "artifact_summary": "",
                },
            ]
        )

        with patch.object(terminal_eval, "_run_task", side_effect=lambda *args, **kwargs: next(fake_results)):
            result = terminal_eval._evaluate_on_tasks(
                program_path=__file__,
                tasks=[{"id": "a"}, {"id": "b"}],
                mode="train",
            )

        assert result.metrics["n_done_signal"] == 1.0
        assert result.metrics["n_verified_completions"] == 0.0
        assert result.metrics["verification_rate"] == 0.0
        assert result.metrics["n_verifier_miss"] == 1.0

    def test_evaluate_stage1_returns_zero_on_infra_crash(self):
        probe = EvaluationResult(
            metrics={"combined_score": 0.0, "n_attempted": 0.0, "mean_task_score": 0.0},
            artifacts={"feedback": "Evaluation crashed: docker unavailable"},
        )
        with patch.object(terminal_eval, "_get_tasks", return_value=[{"id": "json-transform"}]), patch.object(
            terminal_eval, "_evaluate_on_tasks", return_value=probe
        ):
            result = terminal_eval.evaluate_stage1(__file__)

        assert result.metrics["combined_score"] == 0.0
        assert result.metrics["stage1_ok"] == 0.0

    def test_evaluate_stage1_returns_probe_bonus_on_success(self):
        probe = EvaluationResult(
            metrics={
                "combined_score": 1.0,
                "n_attempted": 1.0,
                "mean_task_score": 1.0,
                "eval_wall_s": 2.0,
            },
            artifacts={"feedback": "all good"},
        )
        with patch.object(terminal_eval, "_get_tasks", return_value=[{"id": "json-transform"}]), patch.object(
            terminal_eval, "_evaluate_on_tasks", return_value=probe
        ):
            result = terminal_eval.evaluate_stage1(__file__)

        # score = mean_task_score * 0.1 = 1.0 * 0.1 = 0.1
        assert result.metrics["combined_score"] == 0.1
        assert result.metrics["stage1_ok"] == 1.0
        assert result.metrics["stage1_task_score"] == 1.0


class TestNativeSeed:
    def test_run_pipeline_executes_native_tool_flow(self, tmp_path):
        responses = iter(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_write",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps(
                                    {
                                        "path": (tmp_path / "answer.txt").as_posix(),
                                        "content": "done",
                                    }
                                ),
                            },
                        }
                    ],
                },
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_verify",
                            "type": "function",
                            "function": {
                                "name": "execute_commands",
                                "arguments": json.dumps(
                                    {
                                        "analysis": "Verify the file exists and contains the expected text.",
                                        "plan": "Run a concrete shell check.",
                                        "intent": "verify",
                                        "commands": [
                                            {
                                                "cmd": f"test -f '{(tmp_path / 'answer.txt').as_posix()}' && grep -q done '{(tmp_path / 'answer.txt').as_posix()}'",
                                                "timeout_s": 2,
                                            }
                                        ],
                                    }
                                ),
                            },
                        }
                    ],
                },
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_complete_1",
                            "type": "function",
                            "function": {"name": "task_complete", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_complete_2",
                            "type": "function",
                            "function": {"name": "task_complete", "arguments": "{}"},
                        }
                    ],
                },
            ]
        )

        with patch.object(terminal_seed, "call_llm_tools", side_effect=lambda *args, **kwargs: next(responses)):
            result = terminal_seed.run_pipeline("Create an output file", work_dir=str(tmp_path))

        assert result["done_signal"] is True
        assert result["num_agents"] == terminal_seed.SEED_NUM_AGENTS
        assert result["total_steps"] >= 2
        assert (tmp_path / "answer.txt").read_text() == "done"
        tool_names = [entry["tool"] for entry in result["tool_call_log"]]
        assert tool_names[0] == "write_file"
        assert tool_names.count("execute_commands") >= 1

    def test_normalize_tool_calls_parses_native_tools(self):
        parsed = terminal_seed._normalize_tool_calls(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_one",
                        "type": "function",
                        "function": {
                            "name": "execute_commands",
                            "arguments": json.dumps(
                                {
                                    "analysis": "Inspect",
                                    "plan": "Run pwd",
                                    "intent": "inspect",
                                    "commands": [{"cmd": "pwd"}],
                                }
                            ),
                        },
                    },
                    {
                        "id": "call_two",
                        "type": "function",
                        "function": {"name": "task_complete", "arguments": "{}"},
                    },
                ],
            }
        )

        assert parsed.task_complete is True
        assert [call.name for call in parsed.tool_calls] == ["execute_commands", "task_complete"]
