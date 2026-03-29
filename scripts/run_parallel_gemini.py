#!/usr/bin/env python3
"""
Run 24 parallel experiments: 4 benchmarks × 2 algorithms (adaevolve, evox) × 3 seeds.
Model: gemini-3-pro-preview via Gemini API.
Iterations: 500, checkpoint_interval: 10.
"""

import os
import subprocess
import sys
import yaml
import copy
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GEMINI_API_KEY = "AIzaSyA8k2WiJl07tJUJWrj6os6S_Xz98PFEGWs"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-3-pro-preview"
ITERATIONS = 500
CHECKPOINT_INTERVAL = 10
SEEDS = [42, 123, 456]

BENCHMARKS = {
    # cloudcast requires HuggingFace dataset files (profiles/cost.csv,
    # examples/config/*.json) which must be downloaded via download_dataset.sh.
    # Skip if data is unavailable.
    # "cloudcast": {
    #     "initial_program": "benchmarks/ADRS/cloudcast/initial_program.py",
    #     "evaluator": "benchmarks/ADRS/cloudcast/evaluator/evaluator.py",
    #     "config": "benchmarks/ADRS/cloudcast/config.yaml",
    # },
    "prism": {
        "initial_program": "benchmarks/ADRS/prism/initial_program.py",
        "evaluator": "benchmarks/ADRS/prism/evaluator/evaluator.py",
        "config": "benchmarks/ADRS/prism/config.yaml",
    },
    "heilbronn_convex_13": {
        "initial_program": "benchmarks/math/heilbronn_convex/13/initial_program.py",
        "evaluator": "benchmarks/math/heilbronn_convex/13/evaluator/evaluator.py",
        "config": "benchmarks/math/heilbronn_convex/13/config.yaml",
    },
    "signal_processing": {
        "initial_program": "benchmarks/math/signal_processing/initial_program.py",
        "evaluator": "benchmarks/math/signal_processing/evaluator/evaluator.py",
        "config": "benchmarks/math/signal_processing/config.yaml",
    },
}

ALGORITHMS = ["adaevolve", "evox"]


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def build_merged_config(benchmark_name, benchmark_cfg_path, algorithm, seed):
    """Merge benchmark config with algorithm-specific defaults and Gemini LLM settings."""
    benchmark_data = load_yaml(benchmark_cfg_path)
    algo_template = load_yaml(REPO_ROOT / "configs" / f"{algorithm}.yaml")

    # Start from algo template (has search-specific params), then overlay benchmark settings
    merged = copy.deepcopy(algo_template)

    # Override search type
    if "search" not in merged:
        merged["search"] = {}
    merged["search"]["type"] = algorithm

    # Take prompt/system_message from benchmark
    if "prompt" in benchmark_data:
        merged["prompt"] = benchmark_data["prompt"]

    # Take evaluator settings from benchmark (keep algo defaults as fallback)
    if "evaluator" in benchmark_data:
        merged.setdefault("evaluator", {})
        merged["evaluator"].update(benchmark_data["evaluator"])

    # LLM: Gemini settings
    merged["llm"] = {
        "models": [{"name": GEMINI_MODEL, "weight": 1.0}],
        "api_base": GEMINI_API_BASE,
        "api_key": GEMINI_API_KEY,
        "temperature": 0.7,
        "max_tokens": 32000,
        "timeout": 600,
    }

    # Global settings
    merged["max_iterations"] = ITERATIONS
    merged["checkpoint_interval"] = CHECKPOINT_INTERVAL
    merged["random_seed"] = seed

    # Preserve diff-based generation and solution length from benchmark
    merged["diff_based_generation"] = benchmark_data.get("diff_based_generation", True)
    merged["max_solution_length"] = benchmark_data.get("max_solution_length", 60000)
    merged["language"] = benchmark_data.get("language", "python")

    return merged


def main():
    config_dir = REPO_ROOT / "outputs" / "run_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    log_dir = REPO_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    run_descriptions = []

    for benchmark_name, benchmark_info in BENCHMARKS.items():
        for algorithm in ALGORITHMS:
            for run_idx, seed in enumerate(SEEDS, start=1):
                # Build config
                merged = build_merged_config(
                    benchmark_name,
                    REPO_ROOT / benchmark_info["config"],
                    algorithm,
                    seed,
                )

                # Save merged config
                config_filename = f"{benchmark_name}_{algorithm}_seed{seed}.yaml"
                config_path = config_dir / config_filename
                save_yaml(merged, config_path)

                # Output directory - clearly named
                output_dir = (
                    REPO_ROOT
                    / "outputs"
                    / f"{benchmark_name}__{algorithm}__run{run_idx}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                # Log file
                log_file = log_dir / f"{benchmark_name}__{algorithm}__run{run_idx}.log"

                # Build command
                cmd = [
                    "uv", "run", "skydiscover-run",
                    str(REPO_ROOT / benchmark_info["initial_program"]),
                    str(REPO_ROOT / benchmark_info["evaluator"]),
                    "--config", str(config_path),
                    "--search", algorithm,
                    "--iterations", str(ITERATIONS),
                    "--output", str(output_dir),
                ]

                desc = f"{benchmark_name}__{algorithm}__run{run_idx} (seed={seed})"
                print(f"[LAUNCH] {desc}")
                print(f"  output: {output_dir}")
                print(f"  log:    {log_file}")
                print(f"  cmd:    {' '.join(cmd)}")
                print()

                env = os.environ.copy()
                env["GEMINI_API_KEY"] = GEMINI_API_KEY
                env["OPENAI_API_KEY"] = GEMINI_API_KEY  # some backends need this
                env["OPENAI_API_BASE"] = GEMINI_API_BASE

                with open(log_file, "w") as lf:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(REPO_ROOT),
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )

                processes.append((proc, desc, log_file))
                run_descriptions.append(desc)

    print(f"\n{'='*60}")
    print(f"Launched {len(processes)} parallel runs.")
    print(f"Logs: {log_dir}/")
    print(f"Outputs: {REPO_ROOT}/outputs/")
    print(f"{'='*60}\n")

    # Wait for all processes
    failed = []
    for proc, desc, log_file in processes:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"[{status}] {desc}")
        if rc != 0:
            failed.append((desc, log_file, rc))

    print(f"\n{'='*60}")
    print(f"All {len(processes)} runs complete.")
    if failed:
        print(f"\nFAILED RUNS ({len(failed)}):")
        for desc, log_file, rc in failed:
            print(f"  {desc} -> exit {rc}, log: {log_file}")
    else:
        print("All runs succeeded!")
    print(f"{'='*60}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
