#!/usr/bin/env python3
"""Generate reproducible metadata-balanced TB2 split manifests.

The output manifests are plain text files with one task id per line:
  - tb2_balanced_train.txt
  - tb2_balanced_dev.txt
  - tb2_balanced_test.txt

These can then be activated with:
  TBENCH_TASK_ROOT=/path/to/tb2_harbor_wrapped
  TBENCH_SPLIT_PROFILE=tb2_balanced
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path


def parse_task_metadata(task_dir: Path) -> dict:
    task_toml = task_dir / "task.toml"
    text = task_toml.read_text(encoding="utf-8", errors="replace")

    def extract(pattern: str) -> str:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    tags_match = re.search(r"tags\s*=\s*\[([^\]]*)\]", text)
    tags = re.findall(r'"([^"]+)"', tags_match.group(1)) if tags_match else []
    return {
        "difficulty": extract(r'difficulty\s*=\s*"([^"]+)"') or "unknown",
        "category": extract(r'category\s*=\s*"([^"]+)"') or "unknown",
        "tags": tags,
    }


def discover_tasks(root: Path) -> list[dict]:
    tasks = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (task_dir / "task.toml").exists():
            continue
        if not (task_dir / "instruction.md").exists():
            continue
        if not (task_dir / "tests" / "test.sh").exists():
            continue
        tasks.append(
            {
                "id": task_dir.name,
                "task_dir": task_dir,
                "metadata": parse_task_metadata(task_dir),
            }
        )
    return tasks


def balanced_split(tasks: list[dict], train_ratio: float, dev_ratio: float, seed: int) -> dict:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for task in tasks:
        meta = task["metadata"]
        key = (meta.get("category", "unknown"), meta.get("difficulty", "unknown"))
        buckets[key].append(task)

    splits = {"train": [], "dev": [], "test": []}
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
        splits["train"].extend(ordered[:n_train])
        splits["dev"].extend(ordered[n_train : n_train + n_dev])
        splits["test"].extend(ordered[n_train + n_dev :])

    for split in splits:
        splits[split] = sorted(splits[split], key=lambda item: item["id"])
    return splits


def write_manifest(path: Path, tasks: list[dict]) -> None:
    lines = [task["id"] for task in tasks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TB2 balanced split manifests")
    parser.add_argument(
        "--task-root",
        default="/home/mertcemri/tb2_harbor_wrapped",
        help="Wrapped TB2 task root",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "splits"),
        help="Output directory for manifest files",
    )
    parser.add_argument("--profile", default="tb2_balanced", help="Manifest profile prefix")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_root = Path(args.task_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(task_root)
    splits = balanced_split(tasks, train_ratio=args.train_ratio, dev_ratio=args.dev_ratio, seed=args.seed)
    for split, split_tasks in splits.items():
        manifest = output_dir / f"{args.profile}_{split}.txt"
        write_manifest(manifest, split_tasks)
        print(f"{split}: {len(split_tasks)} -> {manifest}")


if __name__ == "__main__":
    main()
