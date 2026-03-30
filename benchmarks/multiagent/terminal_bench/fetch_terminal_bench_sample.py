#!/usr/bin/env python3
"""Materialize terminal-bench-sample@2.0 into a local Harbor task root."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "multiagent" / "terminal_bench" / "tasksets" / "terminal-bench-sample-2.0"
REGISTRY_REPO = "https://github.com/laude-institute/terminal-bench-2-0-sample.git"
REGISTRY_COMMIT = "7e917f35c281188532772312d4ad91ca9274febc"
REGISTRY_VERSION = "2.0"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _required_paths(task_dir: Path) -> Iterable[Path]:
    return (
        task_dir / "instruction.md",
        task_dir / "tests" / "test.sh",
        task_dir / "environment" / "Dockerfile",
    )


def _is_harbor_task_dir(task_dir: Path) -> bool:
    return task_dir.is_dir() and all(path.exists() for path in _required_paths(task_dir))


def _load_registry(repo_dir: Path) -> dict:
    registry = json.loads((repo_dir / "registry.json").read_text(encoding="utf-8"))
    if isinstance(registry, dict) and "datasets" in registry:
        entries = registry["datasets"]
    elif isinstance(registry, list):
        entries = registry
    else:
        entries = [registry]
    for entry in entries:
        if entry.get("name") == "terminal-bench-sample" and entry.get("version") == REGISTRY_VERSION:
            return entry
    raise RuntimeError(f"Could not find terminal-bench-sample@{REGISTRY_VERSION} in registry.json")


def materialize(output_dir: Path, force: bool = False) -> Path:
    if output_dir.exists():
        if not force:
            raise RuntimeError(f"Output already exists: {output_dir}. Use --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tb2sample-fetch-") as tmp:
        tmp_dir = Path(tmp)
        repo_dir = tmp_dir / "source"
        _run(["git", "clone", REGISTRY_REPO, str(repo_dir)])
        _run(["git", "checkout", REGISTRY_COMMIT], cwd=repo_dir)

        registry_entry = _load_registry(repo_dir)
        sample_root = repo_dir / "sample"
        tasks = registry_entry.get("tasks", [])
        if len(tasks) != 10:
            raise RuntimeError(f"Expected 10 tasks in registry, found {len(tasks)}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            task_name = task["name"]
            task_path = repo_dir / task["path"]
            if not _is_harbor_task_dir(task_path):
                raise RuntimeError(f"Task {task_name} is missing Harbor-required files at {task_path}")
            shutil.copytree(task_path, output_dir / task_name)

        materialized = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
        expected = sorted(task["name"] for task in tasks)
        if materialized != expected:
            raise RuntimeError(f"Materialized tasks {materialized} do not match expected {expected}")
        if not sample_root.exists():
            raise RuntimeError("Source repository did not contain a sample/ directory")

    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch terminal-bench-sample@2.0 into a local task root")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Where to materialize the Harbor tasks")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    materialize(output_dir, force=args.force)
    print(f"Materialized terminal-bench-sample@{REGISTRY_VERSION} into {output_dir}")
    print(f"Harbor reference command: uvx harbor run -d terminal-bench-sample@{REGISTRY_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
