"""Shared utilities for context builders."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class TemplateManager:
    """Loads .txt templates from one or more directories.

    Directories are processed in order; later directories override
    templates with the same name from earlier ones.
    """

    def __init__(self, *directories: Optional[str]):
        """
        Initializes the TemplateManager with the given directories.
        If there are multiple directories, the templates from the later directories will override
        the templates from the earlier directories.
        """
        self.templates: dict[str, str] = {}
        for d in directories:
            if d:
                path = Path(d)
                if path.exists():
                    self._load_from_directory(path)

    def _load_from_directory(self, directory: Path) -> None:
        for txt_file in directory.glob("*.txt"):
            with open(txt_file, "r") as f:
                self.templates[txt_file.stem] = f.read()

    def get_template(self, name: str) -> str:
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]


def prog_attr(program: Any, key: str, default: Any = "") -> Any:
    """Read an attribute from a Program object or a plain dict."""
    if hasattr(program, key):
        return getattr(program, key)
    if isinstance(program, dict):
        return program.get(key, default)
    return default


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    head = max_len // 2
    tail = max_len - head
    return text[:head] + "\n... (truncated) ...\n" + text[-tail:]


def summarize_artifacts(
    artifacts: Optional[Dict[str, Any]],
    *,
    max_chars: int = 1200,
    max_keys: int = 6,
) -> Dict[str, Any]:
    """Build a deterministic bounded summary of evaluator artifacts."""
    if not artifacts:
        return {}

    keys = sorted(str(key) for key in artifacts.keys())[:max_keys]
    summary: Dict[str, Any] = {"artifact_keys": keys}

    feedback = artifacts.get("feedback")
    if feedback:
        summary["artifact_feedback_summary"] = _truncate_text(str(feedback).strip(), max_chars)

    failure_taxonomy = artifacts.get("failure_taxonomy")
    if failure_taxonomy:
        try:
            parsed = failure_taxonomy
            if isinstance(failure_taxonomy, str):
                parsed = json.loads(failure_taxonomy)
            if isinstance(parsed, dict):
                items = ", ".join(f"{k}={v}" for k, v in sorted(parsed.items()))
                summary["artifact_failure_summary"] = _truncate_text(items, max_chars)
            else:
                summary["artifact_failure_summary"] = _truncate_text(str(failure_taxonomy), max_chars)
        except Exception:
            summary["artifact_failure_summary"] = _truncate_text(str(failure_taxonomy), max_chars)
    else:
        failure_parts = []
        for key in ("error", "stderr", "test_stderr", "stdout", "test_stdout"):
            value = artifacts.get(key)
            if value:
                failure_parts.append(f"{key}: {str(value).strip()}")
        if failure_parts:
            summary["artifact_failure_summary"] = _truncate_text("\n".join(failure_parts), max_chars)

    per_task = artifacts.get("per_task")
    if per_task:
        try:
            parsed_tasks = per_task
            if isinstance(per_task, str):
                parsed_tasks = json.loads(per_task)
            if isinstance(parsed_tasks, list):
                failures = [
                    str(item.get("task_id", "unknown"))
                    for item in parsed_tasks
                    if not item.get("completed")
                ][:5]
                if failures:
                    summary["artifact_task_summary"] = "failed_tasks: " + ", ".join(failures)
        except Exception:
            summary["artifact_task_summary"] = _truncate_text(str(per_task), max_chars)

    return summary


def summarize_program_artifacts(
    program: Any,
    *,
    max_chars: int = 1200,
    max_keys: int = 6,
) -> Dict[str, Any]:
    return summarize_artifacts(
        prog_attr(program, "artifacts", {}) or {},
        max_chars=max_chars,
        max_keys=max_keys,
    )


def format_artifact_summary(
    artifact_summary: Dict[str, Any],
    *,
    heading: str = "##",
) -> str:
    if not artifact_summary:
        return ""
    sections = []
    if artifact_summary.get("artifact_keys"):
        sections.append(
            f"{heading} Artifact Keys\n" + ", ".join(str(k) for k in artifact_summary["artifact_keys"])
        )
    if artifact_summary.get("artifact_failure_summary"):
        sections.append(
            f"{heading} Artifact Failure Summary\n{artifact_summary['artifact_failure_summary']}"
        )
    if artifact_summary.get("artifact_feedback_summary"):
        sections.append(
            f"{heading} Artifact Feedback Summary\n{artifact_summary['artifact_feedback_summary']}"
        )
    if artifact_summary.get("artifact_task_summary"):
        sections.append(
            f"{heading} Artifact Task Summary\n{artifact_summary['artifact_task_summary']}"
        )
    if not sections:
        return ""
    return "\n" + "\n\n".join(sections) + "\n"


def format_artifacts(program: Any, heading: str = "##", max_len: int = 2000) -> str:
    """Format evaluator artifacts (e.g. feedback) into markdown sections."""
    artifacts = prog_attr(program, "artifacts", None)
    if not artifacts:
        return ""
    sections = []
    for key, value in artifacts.items():
        if value is None:
            continue
        text = str(value)
        if len(text) > max_len:
            text = text[:max_len] + "\n... (truncated)"
        if key == "feedback":
            sections.append(f"{heading} Evaluator Feedback\n{text}")
        else:
            sections.append(f"{heading} {key}\n{text}")
    if not sections:
        return ""
    return "\n" + "\n\n".join(sections) + "\n"
