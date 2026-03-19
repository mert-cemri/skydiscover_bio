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


class TestBundledHarborTasks:
    def test_discovers_bundled_train_and_test_tasks(self):
        with patch.object(terminal_eval, "_TASK_CACHE", None):
            train_ids = [task["id"] for task in terminal_eval._get_tasks("train")]
            test_ids = [task["id"] for task in terminal_eval._get_tasks("test")]

        assert train_ids == ["git_init_commit", "mkdir_nested", "process_json"]
        assert test_ids == ["compress_archive", "multi_step_pipeline", "permissions_report"]

    def test_choose_stage1_prefers_simple_probe_task(self):
        tasks = [
            {"id": "git_init_commit"},
            {"id": "process_json"},
            {"id": "mkdir_nested"},
        ]
        chosen = terminal_eval._choose_stage1_task(tasks)
        assert chosen["id"] == "mkdir_nested"


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
        assert result.metrics["combined_score"] == 0.515
        assert "Terminal bench train evaluation summary:" in result.artifacts["feedback"]
        per_task = json.loads(result.artifacts["per_task"])
        assert per_task[0]["task_id"] == "a"
        assert per_task[1]["artifact_summary"] == "stderr: fail"

    def test_evaluate_stage1_returns_zero_on_infra_crash(self):
        probe = EvaluationResult(
            metrics={"combined_score": 0.0, "n_attempted": 0.0, "mean_task_score": 0.0},
            artifacts={"feedback": "Evaluation crashed: docker unavailable"},
        )
        with patch.object(terminal_eval, "_get_tasks", return_value=[{"id": "mkdir_nested"}]), patch.object(
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
        with patch.object(terminal_eval, "_get_tasks", return_value=[{"id": "mkdir_nested"}]), patch.object(
            terminal_eval, "_evaluate_on_tasks", return_value=probe
        ):
            result = terminal_eval.evaluate_stage1(__file__)

        assert result.metrics["combined_score"] == 0.1
        assert result.metrics["stage1_ok"] == 1.0
        assert result.metrics["stage1_task_score"] == 1.0


class TestMultiAgentSeed:
    def test_run_pipeline_executes_multi_agent_handoffs(self, tmp_path):
        responses = iter(
            [
                """
<analysis>Need to inspect the workspace before changing anything.</analysis>
<memory_update>Planner delegated inspection.</memory_update>
<next_agent>researcher</next_agent>
<commands></commands>
""",
                """
<analysis>The workspace exists and is ready.</analysis>
<memory_update>Research confirms a writable workspace.</memory_update>
<next_agent>executor</next_agent>
<commands>
TOOL: list_dir | {"path": "."}
</commands>
""",
                f"""
<analysis>Create the required output file now.</analysis>
<memory_update>Executor is creating the output artifact.</memory_update>
<next_agent>verifier</next_agent>
<commands>
TOOL: write_file | {{"path": "{(tmp_path / 'answer.txt').as_posix()}", "content": "done"}}
</commands>
""",
                f"""
<analysis>The expected file exists and can be checked.</analysis>
<memory_update>Verification succeeded.</memory_update>
<next_agent>verifier</next_agent>
<commands>
TOOL: read_file | {{"path": "{(tmp_path / 'answer.txt').as_posix()}"}}
</commands>
<task_complete>true</task_complete>
""",
            ]
        )

        with patch.object(terminal_seed, "call_llm_multi", side_effect=lambda *args, **kwargs: next(responses)):
            result = terminal_seed.run_pipeline("Create an output file", work_dir=str(tmp_path))

        assert result["done_signal"] is True
        assert result["num_agents"] == len(terminal_seed.AGENT_SPECS)
        assert result["total_steps"] == 3
        assert (tmp_path / "answer.txt").read_text() == "done"
        assert [entry["agent"] for entry in result["tool_call_log"]] == [
            "researcher",
            "executor",
            "verifier",
        ]

    def test_filter_tool_calls_enforces_role_permissions(self):
        planner = next(spec for spec in terminal_seed.AGENT_SPECS if spec.name == "planner")
        calls = [
            terminal_seed.ToolCall(name="write_file", args={"path": "/tmp/x", "content": "bad"}),
            terminal_seed.ToolCall(name="run_shell", args={"cmd": "pwd"}),
        ]

        filtered = terminal_seed.filter_tool_calls(planner, calls)

        assert [call.name for call in filtered] == ["run_shell"]
