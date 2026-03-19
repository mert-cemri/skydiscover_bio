# Terminal Bench

Harbor-backed multi-agent terminal scaffold benchmark.

## What Is Optimized

The seed keeps only the primitive runtime fixed:
- `call_llm_multi(...)`
- `run_shell`, `read_file`, `write_file`, `list_dir`
- `run_agentic(task, work_dir)`
- `run_pipeline(task, work_dir)` return schema

Nearly the entire scaffold lives inside the EVOLVE block:
- agent roster and count
- role prompts and tool permissions
- handoff graph and scheduler
- shared memory and summarization
- parsing and observation formatting
- retry, verification, and completion logic

## Evaluation

The evaluator runs Harbor-format Docker tasks instead of host-local shell scripts.

Default behavior:
- uses bundled tasks under `tasks/train` and `tasks/test`

Optional external mode:
- set `TBENCH_TASK_ROOT=/path/to/harbor/taskset`
- if that root has `train/` and `test/`, those are used directly
- otherwise tasks are hash-split with `TBENCH_TRAIN_RATIO`

## Run

```bash
uv sync
export OPENAI_API_KEY="..."

uv run skydiscover-run \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  benchmarks/multiagent/terminal_bench/evaluator.py \
  --config benchmarks/multiagent/terminal_bench/config.yaml
```

Use an external Harbor task collection:

```bash
TBENCH_TASK_ROOT=/path/to/tasks \
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
