"""Tests for artifact-driven prompt, feedback, and monitor behavior."""

from skydiscover.config import AIFeedbackConfig, Config
from skydiscover.context_builder.ai_feedback import AIFeedbackReader
from skydiscover.context_builder.default.builder import DefaultContextBuilder
from skydiscover.context_builder.human_feedback import HumanFeedbackReader
from skydiscover.extras.monitor.callback import create_monitor_callback
from skydiscover.search.base_database import Program


class _FakeServer:
    def __init__(self):
        self.max_solution_length = 1000
        self.artifact_summary_max_chars = 500
        self.artifact_summary_max_keys = 5
        self.events = []

    def push_event(self, event):
        self.events.append(event)


class _FakeDatabase:
    def __init__(self, programs):
        self.programs = {program.id: program for program in programs}
        self.best_program_id = programs[0].id if programs else None

    def get(self, program_id):
        return self.programs.get(program_id)

    def get_best_program(self):
        return self.programs.get(self.best_program_id)


class TestArtifactDrivenUsage:
    def test_default_builder_renders_artifact_summaries(self):
        config = Config(language="python")
        builder = DefaultContextBuilder(config)
        current = Program(
            id="p1",
            solution="print('hello')",
            metrics={"combined_score": 0.1},
            artifacts={
                "feedback": "timed out on 3 tasks",
                "failure_taxonomy": '{"timeout": 3, "task_failure": 1}',
            },
        )
        context_program = Program(
            id="p2",
            solution="print('context')",
            metrics={"combined_score": 0.2},
            artifacts={"stderr": "bad verifier command"},
        )
        prompt = builder.build_prompt(
            current_program=current,
            context={
                "program_metrics": current.metrics,
                "other_context_programs": {"Related": [context_program]},
                "previous_programs": [current],
                "errors": [
                    {
                        "solution": "print('bad')",
                        "metadata": {"error": "Evaluation failed", "attempt_number": 1},
                        "artifacts": {"stderr": "docker exec timed out after 60s"},
                    }
                ],
            },
        )
        assert "Artifact Failure Summary" in prompt["user"]
        assert "timeout=3" in prompt["user"]
        assert "docker exec timed out after 60s" in prompt["user"]
        assert "bad verifier command" in prompt["user"]

    def test_human_feedback_reader_exposes_artifact_summary(self, tmp_path):
        reader = HumanFeedbackReader(str(tmp_path / "feedback.md"))
        reader.set_current_prompt("System prompt")
        reader.set_artifact_summary({"artifact_failure_summary": "timed out on 3 tasks"})
        current_prompt = reader.get_current_prompt()
        assert "System prompt" in current_prompt
        assert "timed out on 3 tasks" in current_prompt

    def test_ai_feedback_context_includes_artifact_summary(self):
        config = AIFeedbackConfig(enabled=True, include_artifact_summary=True)
        reader = AIFeedbackReader(config=config, call_llm_fn=lambda **kwargs: "No changes needed.")
        context = reader._build_context(
            {
                "solution_score_summary": {"best": 0.1, "q75": 0.1, "q50": 0.1, "q25": 0.1},
                "latest_artifact_summary": "timed out on 3 tasks",
            },
            iteration=7,
        )
        assert "Latest evaluator artifact summary" in context
        assert "timed out on 3 tasks" in context
        reader.stop()

    def test_monitor_callback_includes_artifact_summary(self):
        program = Program(
            id="p1",
            solution="print('hello')",
            metrics={"combined_score": 0.1},
            artifacts={"failure_taxonomy": '{"timeout": 2}', "feedback": "timed out"},
        )
        server = _FakeServer()
        database = _FakeDatabase([program])
        callback = create_monitor_callback(server, database, start_time=0.0)
        callback(program, iteration=1)
        event = server.events[-1]
        assert event["program"]["artifact_summary"]["artifact_keys"] == ["failure_taxonomy", "feedback"]
        assert "timeout=2" in event["program"]["artifact_summary"]["artifact_failure_summary"]
