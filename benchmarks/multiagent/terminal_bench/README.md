# Terminal Bench

Harbor-backed terminal scaffold benchmark with a stronger native-tool seed harness.

## What Is Optimized

The seed now includes a stronger fixed runtime:
- `call_llm_multi(...)` and native `call_llm_tools(...)`
- an in-container interactive shell session
- file and search helpers
- `run_agentic(task, work_dir)`
- `run_pipeline(task, work_dir)` return schema

Nearly the entire harness policy still lives inside the EVOLVE block:
- tool schemas
- memory and summarization
- seed profile (`single_agent_native` vs `two_phase`)
- verification and completion policy
- retry and recovery logic

## Evaluation

The evaluator runs Harbor-format Docker tasks instead of host-local shell scripts.

Default behavior:
- uses bundled tasks under `tasks/train` and `tasks/test` for cheap smoke gating

Optional external mode:
- set `TBENCH_TASK_ROOT=/path/to/harbor/taskset`
- if that root has `train/` and `test/`, those are used directly
- otherwise tasks can be split either by hash or via balanced metadata buckets

Recommended protocol:
- Tier 1: bundled TB1-style smoke gate
- Tier 2: `terminal-bench-sample@2.0` train/dev for real optimization and seed/model selection
- Tier 3: official full TB2 only for final held-out evaluation

Materialize the Harbor sample:

```bash
uv run python benchmarks/multiagent/terminal_bench/fetch_terminal_bench_sample.py
```

This creates:

- `benchmarks/multiagent/terminal_bench/tasksets/terminal-bench-sample-2.0`

Harbor-sample train/dev mode:
- set `TBENCH_TASK_ROOT=benchmarks/multiagent/terminal_bench/tasksets/terminal-bench-sample-2.0`
- set `TBENCH_SPLIT_PROFILE=tb2sample`
- use `TBENCH_SEED_PROFILE=single_agent_native` or `TBENCH_SEED_PROFILE=two_phase`

Official TB2 held-out mode:
- set `TBENCH_TASK_ROOT=/home/mertcemri/tb2_harbor_wrapped`
- set `TBENCH_SPLIT_PROFILE=tb2_balanced`

## Run

```bash
uv sync
export OPENAI_API_KEY="..."

uv run skydiscover-run \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  benchmarks/multiagent/terminal_bench/evaluator.py \
  --config benchmarks/multiagent/terminal_bench/config.yaml
```

Use the alternative two-phase seed:

```bash
TBENCH_SEED_PROFILE=two_phase \
uv run skydiscover-run \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  benchmarks/multiagent/terminal_bench/evaluator.py \
  --config benchmarks/multiagent/terminal_bench/config.yaml
```

Use the Harbor sample as the train/dev regime:

```bash
uv run python benchmarks/multiagent/terminal_bench/fetch_terminal_bench_sample.py

TBENCH_TASK_ROOT=benchmarks/multiagent/terminal_bench/tasksets/terminal-bench-sample-2.0 \
TBENCH_SPLIT_PROFILE=tb2sample \
uv run skydiscover-run \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  benchmarks/multiagent/terminal_bench/evaluator.py \
  --config benchmarks/multiagent/terminal_bench/config.yaml
```

Use the official TB2 profile only for held-out testing:

```bash
TBENCH_TASK_ROOT=/home/mertcemri/tb2_harbor_wrapped \
TBENCH_SPLIT_PROFILE=tb2_balanced \
TBENCH_SEED_PROFILE=single_agent_native \
uv run skydiscover-run \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  benchmarks/multiagent/terminal_bench/evaluator.py \
  --config benchmarks/multiagent/terminal_bench/config.yaml
```

Held-out evaluation:

```bash
python benchmarks/multiagent/terminal_bench/evaluator.py \
  --test outputs/.../best/best_program.py
```

Explicit split evaluation:

```bash
TBENCH_TASK_ROOT=benchmarks/multiagent/terminal_bench/tasksets/terminal-bench-sample-2.0 \
TBENCH_SPLIT_PROFILE=tb2sample \
uv run python benchmarks/multiagent/terminal_bench/evaluator.py \
  --split dev outputs/.../best/best_program.py
```

Regenerate balanced TB2 manifests:

```bash
uv run python benchmarks/multiagent/terminal_bench/build_tb2_balanced_splits.py
```

Run evaluator-level ablations:

```bash
uv run python benchmarks/multiagent/terminal_bench/run_ablation_suite.py \
  --models gpt-5.2 gpt-5 \
  --seed-profiles single_agent_native two_phase \
  --task-root benchmarks/multiagent/terminal_bench/tasksets/terminal-bench-sample-2.0 \
  --split-profile tb2sample
```
