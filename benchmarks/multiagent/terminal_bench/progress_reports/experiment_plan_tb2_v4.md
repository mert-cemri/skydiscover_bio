# TB2 Multi-Agent Scaffold Optimization — Experiment Plan v4

**Written:** 2026-03-26
**Replaces:** ad-hoc v3 search strategy
**Status:** Ready to execute

---

## Executive Context

The v3 search achieved 0.8332 on the 10-task internal benchmark but only 7.87% (7/89) on full TB2.
The gap is severe and has a well-understood set of causes:

1. **Training distribution mismatch**: v3 trained on a fixed `random.seed(42)` sample from the 36
   TB1-converted train tasks. Those tasks differ substantially from real TB2 in difficulty, toolchain
   diversity, and format.

2. **Acceptance command blind spot**: `_discover_acceptance_hint()` never checked `/app/tests/test.sh`,
   which is exactly where the Harbor RUNNER block injects the scaffold wrapper. Every TB2 task has this
   path but the scaffold was never probing for it.

3. **False completion signals**: `ACCEPTANCE_TOKENS` included `test -f`, `grep -q`, `diff`, and `cmp`,
   which fire trivially on any file-existence or content check. This let the scaffold declare
   `acceptance_verified_since_mutation = True` without running a real test harness.

4. **Timeout blindness on hard tasks**: TB2 `task.toml` files set `timeout_sec=900`. Without
   `TBENCH_MAX_TASK_TIMEOUT`, the HarborEvaluator adopted this per-task, making each failed attempt
   spend up to 15 minutes. The CANDIDATE_TIMEOUT_S cap was being ignored.

5. **apt-get not treated as heavy**: Package installation commands weren't in `HEAVY_COMMAND_TOKENS`,
   so they ran with the standard 25s timeout and failed silently.

6. **Cascade gate too aggressive**: Stage2 threshold of 0.30 required solving 3 out of 10 TB2-style
   tasks before proceeding to stage3. With honest TB2 tasks, even a genuinely improving scaffold
   might not cross 0.30 at early iterations, causing early rejection of good candidates.

**This plan addresses all six root causes before launching the main evolutionary run.**

---

## Code Changes Already Applied

The following changes have been applied to `initial_program.py` and `config.yaml` before any
experimentation begins. These are mandatory pre-conditions for the experiments below.

### Fix 1 — `_discover_acceptance_hint()`: probe `/app/tests/test.sh` first

```python
# Before (missed the Harbor RUNNER wrapper path entirely):
"if [ -f /tests/verify.sh ]; then echo 'bash /tests/verify.sh'; ..."

# After (checks /app/tests/test.sh as the FIRST candidate):
"if [ -f /app/tests/test.sh ]; then echo 'bash /app/tests/test.sh'; "
"elif [ -f /tests/verify.sh ]; then echo 'bash /tests/verify.sh'; ..."
```

**Why this matters**: Every TB2 task wrapped with the RUNNER block has its test harness at
`/app/tests/test.sh`. Probing for this first gives the scaffold the correct acceptance command
on the very first turn, rather than falling back to generic guesses.

### Fix 2 — `ACCEPTANCE_TOKENS`: strip false-positive patterns

```python
# Before (broad, triggers on shell idioms):
ACCEPTANCE_TOKENS = (
    "pytest", "make test", "npm test", "cargo test", "go test", "ctest",
    "verify.sh", "test -f", "test -d", "cmp ", "diff ", "grep -q",
)

# After (only real test runners qualify as acceptance evidence):
ACCEPTANCE_TOKENS = (
    "pytest", "make test", "npm test", "cargo test", "go test", "ctest",
    "verify.sh", "test.sh",
)
```

**Why this matters**: `test -f /app/solution.py` used to satisfy `acceptance_verified_since_mutation`,
causing `task_complete` to fire as `done_signal=True` with score 0.0. This was the primary driver
of the false-completion pattern seen in `query-optimize` and many other TB2 tasks.

### Fix 3 — `HEAVY_COMMAND_TOKENS`: add apt-get

```python
# Added:
"apt-get",
"apt ",
```

**Why this matters**: TB2 tasks frequently require `apt-get install` for compilers, interpreters,
or runtime dependencies. Without this flag, apt-get ran with the default 25s timeout and silently
failed on packages that need 30-120 seconds to install, leaving the environment broken for all
subsequent steps.

### Fix 4 — `ACCEPTANCE_PLAYBOOK`: add `/app/tests/test.sh` as first entry

```python
# Before:
ACCEPTANCE_PLAYBOOK = [
    "bash /tests/verify.sh",
    ...
]

# After:
ACCEPTANCE_PLAYBOOK = [
    "bash /app/tests/test.sh",   # ← new first entry
    "bash /tests/verify.sh",
    ...
]
```

**Why this matters**: When the scaffold decides to brute-force acceptance commands from the playbook,
it now tries the most common TB2 path first, increasing the chance of discovering the working
command within the step budget.

### Config Change — Cascade threshold and iteration count

```yaml
# Before:
max_iterations: 50
cascade_thresholds:
  - 0.05
  - 0.3

# After:
max_iterations: 75
cascade_thresholds:
  - 0.05
  - 0.12
```

**Why**: The 0.30 stage2 threshold was calibrated for easy TB1-converted tasks where the seed already
scored 0.30+. On honest TB2 tasks, the seed scores below 0.10. A threshold of 0.12 means "solve at
least 1-2 tasks out of 10", which is achievable by an improving scaffold without being too permissive.
75 iterations instead of 50 gives AdaEvolve more budget to find paradigm breakthroughs after the
seed improvements start yielding diminishing returns.

---

## Phase 0 — Preflight Smoke (Mandatory Gate)

**Purpose**: Confirm that the fixed seed scaffold can complete the easiest bundled tasks before
investing any GPU/API budget in TB2-scale evaluation. If the smoke fails, fix the seed again.

**What we test**: The 1-task stage1 probe (`json-transform`) plus a manual run of 2-3 more easy
bundled tasks to confirm end-to-end plumbing works.

**Commands**:

```bash
# Single smoke task: stage1 probe only
SOLVER_MODEL=gpt-4o-mini \
TBENCH_MAX_TASK_TIMEOUT=120 \
CANDIDATE_TIMEOUT_S=60 \
uv run python benchmarks/multiagent/terminal_bench/evaluator.py \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  --split train 2>&1 | tail -30

# Optional: run all bundled train tasks with the seed (5-10 min)
SOLVER_MODEL=gpt-4o-mini \
TBENCH_MAX_TASK_TIMEOUT=120 \
CANDIDATE_TIMEOUT_S=90 \
TRAIN_TASKS=999 \
uv run python benchmarks/multiagent/terminal_bench/evaluator.py \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  --all --split train
```

**Gate criteria**:
- `json-transform` must score 1.0 (stage1 probe)
- At least 3 of the 4 known-easy bundled tasks must score 1.0:
  `json-transform`, `git-init-commit`, `fix-git`, `find-replace-text`
- Zero scaffold crashes
- No `done_signal=True` on a task that scored 0.0 (false completion check)

**If gate fails**: Do not proceed. Fix the scaffold bug causing the failure. Re-run smoke.

---

## Phase 1 — Crashprobe on TB2 (Diagnostic, Not a Gate)

**Purpose**: Understand how the fixed seed handles the hardest TB2 tasks that have historically
caused 100% timeout. This is a diagnostic read on robustness, not a hard pass/fail gate.

**Background**: The 3 crashprobe tasks (`sam-cell-seg`, `llm-inference-batching-scheduler`,
`reshard-c4-data`) timed out in every previous run (0/3 three times across three dates).
The fixes to apt-get handling and `/app/tests/test.sh` discovery may help `reshard-c4-data`
and `llm-inference-batching-scheduler`. `sam-cell-seg` will likely still be hard due to SAM's
large model download.

**Commands**:

```bash
# Run the crashprobe split (3 train + 1 dev + 1 test = 5 tasks)
SOLVER_MODEL=gpt-4o-mini \
TBENCH_TASK_ROOT=/home/mertcemri/tb2_harbor_wrapped \
TBENCH_SPLIT_PROFILE=tb2_crashprobe \
TBENCH_MAX_TASK_TIMEOUT=300 \
CANDIDATE_TIMEOUT_S=300 \
uv run python benchmarks/multiagent/terminal_bench/run_hard_mini_suite.py \
  --output /tmp/crashprobe_fixed_seed.json \
  --models gpt-4o-mini \
  --seed-profiles single_agent_native \
  --splits train dev
```

**Or using the evaluator directly**:

```bash
SOLVER_MODEL=gpt-4o-mini \
TBENCH_TASK_ROOT=/home/mertcemri/tb2_harbor_wrapped \
TBENCH_SPLIT_PROFILE=tb2_crashprobe \
TBENCH_MAX_TASK_TIMEOUT=300 \
CANDIDATE_TIMEOUT_S=300 \
TRAIN_TASKS=999 DEV_TASKS=999 \
uv run python benchmarks/multiagent/terminal_bench/evaluator.py \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  --all --split train
```

**Interpret results as**:
- 0/3 (all timeout): Fixes help with acceptance discovery but not with heavy setup. Proceed anyway — AdaEvolve can evolve setup strategies.
- 1/3 (one completes): Real improvement. The acceptance fix is working. Good signal to proceed.
- 2-3/3: Strong improvement. The seed is substantially more capable on hard tasks. Excellent.

**This is NOT a blocking gate.** Even at 0/3, the fixes are theoretically sound and the seed
is better than v3. Proceed to Phase 2 regardless.

---

## Phase 2 — TB2 Sample Ablation (Model / Profile Selection)

**Purpose**: Pick the best (solver model, seed profile) combination for the full AdaEvolve run
using a 10-task ablation on the `tb2sample` materialized taskset.

**Background**: `tb2sample` contains 10 Harbor tasks from TB2:
`build-cython-ext`, `chess-best-move`, `configure-git-webserver`, `fix-code-vulnerability`,
`log-summary-date-ranges`, `polyglot-c-py`, `qemu-alpine-ssh`, `qemu-startup`, `regex-log`,
`sqlite-with-gcov`.

The `configure-git-webserver` task previously had `AttributeError: module 'solution' has no
attribute 'run_agentic'` — this was an infrastructure edge case. With the fixed acceptance path,
it should resolve.

**Commands**:

```bash
# Ablation matrix: 1 solver model × 1 seed profile × 2 splits
TBENCH_TASK_ROOT=benchmarks/multiagent/terminal_bench/tasksets/terminal-bench-sample-2.0 \
TBENCH_SPLIT_PROFILE=tb2sample \
TBENCH_MAX_TASK_TIMEOUT=300 \
uv run python benchmarks/multiagent/terminal_bench/run_ablation_suite.py \
  --output /tmp/ablation_tb2sample_fixed.json \
  --models gpt-4o-mini \
  --seed-profiles single_agent_native \
  --splits train dev \
  --timeout 300

# If gpt-4o (full) is available and budget permits, also run:
TBENCH_TASK_ROOT=benchmarks/multiagent/terminal_bench/tasksets/terminal-bench-sample-2.0 \
TBENCH_SPLIT_PROFILE=tb2sample \
TBENCH_MAX_TASK_TIMEOUT=300 \
uv run python benchmarks/multiagent/terminal_bench/run_ablation_suite.py \
  --output /tmp/ablation_tb2sample_gpt4o.json \
  --models gpt-4o \
  --seed-profiles single_agent_native \
  --splits train dev \
  --timeout 300
```

**Interpret results as**:
- Score on `tb2sample train` is the key signal. Target: ≥ 2/6 (33%) with `gpt-4o-mini`.
- If `gpt-4o` scores meaningfully higher (≥ 2 more tasks), use it as the solver. Otherwise use `gpt-4o-mini` to keep costs manageable across 75 iterations.
- `two_phase` profile has been consistently broken (API 400 errors, all-timeout on hard tasks). Do not use it unless ablation shows a surprise reversal.
- The winning (model, profile) pair from this phase becomes SOLVER_MODEL and TBENCH_SEED_PROFILE for Phase 4.

**Gate criteria (to proceed to Phase 3)**:
- At least 1/6 tasks completed on `tb2sample train` (16% completion rate)
- Zero scaffold crashes (returncode -1 / Python exception in scaffold, not task failure)
- `done_signal` precision > 0% (i.e., at least one `done_signal=True` task also has `score=1.0`)

---

## Phase 3 — Hard-Mini Gate

**Purpose**: Confirm the seed is robust enough on hard tasks before launching the expensive 75-iteration
search. This is the formal go/no-go gate. The previous hard-mini result was 0/6. We need ≥ 1-2 to
proceed confidently.

**Background**: The `tb2_hardmini` profile contains 14 tasks across train/dev/test splits (6/4/4).
These tasks were selected specifically for being hard: heavy compilation, ML setup, complex
verification. They are the clearest stress test of the scaffold's robustness.

**Commands**:

```bash
# Run hard-mini suite with fixed seed
SOLVER_MODEL=gpt-4o-mini \
TBENCH_TASK_ROOT=/home/mertcemri/tb2_harbor_wrapped \
TBENCH_SPLIT_PROFILE=tb2_hardmini \
TBENCH_MAX_TASK_TIMEOUT=420 \
CANDIDATE_TIMEOUT_S=420 \
uv run python benchmarks/multiagent/terminal_bench/run_hard_mini_suite.py \
  --output /tmp/hardmini_fixed_seed.json \
  --models gpt-4o-mini \
  --seed-profiles single_agent_native \
  --splits train dev

# Check against the gate thresholds
uv run python benchmarks/multiagent/terminal_bench/gate_seed_readiness.py \
  /tmp/hardmini_fixed_seed.json \
  --min-completion-rate 0.10 \
  --min-mean-score 0.10 \
  --max-crashes 0.50 \
  --min-verification-rate 0.50
```

**Gate thresholds** (lowered from default values because we are testing a seed before evolution):
- `--min-completion-rate 0.10`: at least 1 task out of 10 (train + dev) must complete
- `--min-mean-score 0.10`: mean score across all attempted tasks ≥ 0.10
- `--max-crashes 0.50`: no more than 50% scaffold crashes (Python exception / missing `run_agentic`)
- `--min-verification-rate 0.50`: at least half of done_signal=True results must have score > 0

**Exit codes from `gate_seed_readiness.py`**:
- 0 = proceed to Phase 4 (evolution)
- 1 = seed still needs hardening — revisit fixes, then re-run Phase 1-3
- 2 = no evaluation data — something is wrong with the run itself

**If gate fails at threshold 0.10**: The seed fixes were insufficient for hard tasks. The likely
remaining issues are:
1. Docker images for heavy tasks take too long to build (not a scaffold issue — pre-build images)
2. Some TB2 tasks have environment-setup scripts that the scaffold cannot run fast enough
3. The PTY shell `MAX_COMMAND_TIMEOUT` of 180s is still too low for some packages

In that case, try increasing `MAX_COMMAND_TIMEOUT` to 300 (inside EVOLVE-BLOCK) and re-run.

---

## Phase 4 — Main AdaEvolve Search (tb2_balanced, 75 iterations)

**Purpose**: Run the full evolutionary search with the fixed seed and correct training distribution.

### Training distribution: `tb2_balanced`

Use the `tb2_balanced` split profile (64 train / 7 dev / 18 test tasks from TB2).
The evaluator samples TRAIN_TASKS=10 at random with seed 42 from the 64 train tasks for each
evaluation call. This is substantially more diverse and harder than the old 36-task TB1 sample.

**Key differences from v3**:
- Training tasks are real TB2 tasks (Harbor-format, diverse toolchains)
- The cascade stage2 threshold is 0.12 instead of 0.30 (more permissive for early iterations)
- TBENCH_MAX_TASK_TIMEOUT=300 prevents per-task timeouts from spiraling to 900s
- The fixed seed has the correct acceptance command probe and tighter completion logic
- 75 iterations instead of 50 gives more budget for paradigm breakthroughs

### Main run command

```bash
SOLVER_MODEL=gpt-4o-mini \
TRAIN_TASKS=10 \
CANDIDATE_TIMEOUT_S=300 \
TBENCH_TASK_ROOT=/home/mertcemri/tb2_harbor_wrapped \
TBENCH_SPLIT_PROFILE=tb2_balanced \
TBENCH_MAX_TASK_TIMEOUT=300 \
nohup uv run skydiscover-run \
  benchmarks/multiagent/terminal_bench/initial_program.py \
  benchmarks/multiagent/terminal_bench/evaluator.py \
  --config benchmarks/multiagent/terminal_bench/config.yaml \
  --output outputs/terminal_bench_adaevolve_v4 \
  > outputs/terminal_bench_adaevolve_v4/run.log 2>&1 &

echo "PID: $!"
```

### Monitoring

```bash
# Tail the run log
tail -f outputs/terminal_bench_adaevolve_v4/run.log

# Watch the live dashboard
# http://localhost:8889/

# Check current best
cat outputs/terminal_bench_adaevolve_v4/best/best_program_info.json | python3 -m json.tool

# Check checkpoint scores
ls outputs/terminal_bench_adaevolve_v4/checkpoints/
cat outputs/terminal_bench_adaevolve_v4/checkpoints/checkpoint_*/adaevolve_metadata.json \
  | python3 -c "import sys,json; [print(json.load(open(f))['global_best_score']) for f in sys.argv[1:]]"
```

### Early stopping criteria

**Stop and examine if any of these occur:**
1. No improvement in global_best_score for 20 consecutive iterations (stagnation)
2. Global best score suddenly drops by > 0.10 (instability / regression)
3. More than 30% of iterations produce scaffold Python exceptions (infra problem)
4. API cost projection exceeds budget

**Do NOT stop early if:**
- Score is slowly improving (even 0.01/iteration is meaningful)
- A paradigm breakthrough fires (identified in logs as "paradigm_breakthrough=True")
- Score temporarily dips on one iteration but recovers the next

### Expected progression

| Iteration range | Expected combined_score | Interpretation |
|---|---|---|
| 0-5 | 0.05 – 0.15 | Seed on real TB2, hard tasks, lower than v3 because TB2 is harder |
| 5-20 | 0.10 – 0.25 | Search begins finding improvement in acceptance/verification logic |
| 20-40 | 0.20 – 0.40 | Stronger solutions emerge, setup strategies improving |
| 40-75 | 0.30 – 0.55+ | Paradigm breakthroughs; best solutions handle 3-5 hard tasks |

These are rough targets. If scores are tracking higher, excellent. If they plateau below 0.15
after 25 iterations, see the troubleshooting section below.

### AdaEvolve configuration details

```yaml
# config.yaml (current state, do not change for this run)
max_iterations: 75
cascade_thresholds: [0.05, 0.12]
search:
  type: adaevolve
  database:
    population_size: 20
    num_islands: 2
    use_dynamic_islands: true
    max_islands: 4
    use_paradigm_breakthrough: true
    paradigm_window_size: 10
    paradigm_improvement_threshold: 0.10
    paradigm_max_uses: 3
    use_unified_archive: true
    decay: 0.9
ai_feedback:
  enabled: true
  model: gpt-5
  interval: 5
```

The AI feedback fires every 5 iterations and will see the TB2-style failures. This is intentional:
the feedback will naturally steer the LLM toward better verification discipline, retry logic,
and setup strategies because those are the dominant failure modes on real TB2 tasks.

### Mid-run dev-split check (optional, at iteration ~35)

At roughly the midpoint, pause the run and evaluate the current best on the dev split to get a
read on generalization before the run completes:

```bash
# Evaluate current best on dev split (7 tasks, ~15 min)
SOLVER_MODEL=gpt-4o-mini \
TBENCH_TASK_ROOT=/home/mertcemri/tb2_harbor_wrapped \
TBENCH_SPLIT_PROFILE=tb2_balanced \
TBENCH_MAX_TASK_TIMEOUT=300 \
CANDIDATE_TIMEOUT_S=300 \
DEV_TASKS=999 \
uv run python benchmarks/multiagent/terminal_bench/evaluator.py \
  outputs/terminal_bench_adaevolve_v4/best/best_program.py \
  --split dev \
  --output /tmp/v4_midrun_dev_eval.json
```

If dev score is tracking with train score (within ~0.10), the scaffold is generalizing. If dev
score is much lower (< 50% of train), consider whether the training sample is being overfit.

---

## Phase 5 — Final Held-Out TB2 Evaluation

**Purpose**: Produce the definitive result on the unseen test split. Run only once, on the best
scaffold from Phase 4, with a frozen config. Do NOT tune anything after seeing test results.

**When to run**: After Phase 4 completes all 75 iterations and the best program is stable.

### Using the evaluator test split (18 tasks)

```bash
SOLVER_MODEL=gpt-4o-mini \
TBENCH_TASK_ROOT=/home/mertcemri/tb2_harbor_wrapped \
TBENCH_SPLIT_PROFILE=tb2_balanced \
TBENCH_MAX_TASK_TIMEOUT=300 \
CANDIDATE_TIMEOUT_S=300 \
TEST_TASKS=999 \
uv run python benchmarks/multiagent/terminal_bench/evaluator.py \
  outputs/terminal_bench_adaevolve_v4/best/best_program.py \
  --split test \
  --output outputs/tb2_eval_v4_final/test_eval.json
```

### Using the parallel runner for full 89-task TB2 evaluation

```bash
# Full TB2 evaluation (89 tasks, ~10 min with 50 workers)
python /tmp/run_tb2_parallel.py \
  --program outputs/terminal_bench_adaevolve_v4/best/best_program.py \
  --task-root /home/mertcemri/tb2_harbor_wrapped \
  --output-dir outputs/tb2_eval_v4_parallel \
  --max-workers 50 \
  --timeout 300

# Summary
cat outputs/tb2_eval_v4_parallel/summary.json | python3 -m json.tool
```

**Report the following numbers**:
- Completion rate on held-out test split (18 tasks): `N/18`
- Completion rate on full 89-task TB2: `N/89`
- Mean score, efficiency bonus, verification bonus
- List of passed/failed tasks for qualitative analysis
- Comparison to v3 baseline (7/89 = 7.87%)

**Target**: ≥ 15/89 (≥ 17%) on full TB2. This represents roughly a 2× improvement over v3.
A result of ≥ 20/89 (≥ 22%) would be excellent. Below 10/89 indicates the training distribution
or cascade gate needs further work.

---

## Troubleshooting Guide

### Scenario: Score plateau below 0.10 after 20 iterations

**Diagnosis**: The stage1 smoke gate (threshold 0.05) is too easy — it only needs the json-transform
probe task to score 0.005 (stage1 gives a 0.1× multiplier). Meanwhile, stage2 threshold of 0.12 on
10 hard TB2 tasks is rejecting too many candidates that are actually improving in subtle ways.

**Fix**: Lower stage2 threshold further to 0.08, or examine the best scaffold at iteration 20 manually
to understand which tasks it is solving and which it is failing.

### Scenario: Many `AttributeError: module 'solution' has no attribute 'run_agentic'`

**Diagnosis**: The scaffold file is being imported but `run_agentic` is not defined at module level —
typically caused by a Python syntax error inside the EVOLVE block that prevents the module from
fully loading.

**Fix**:
```bash
python3 -c "import ast; ast.parse(open('outputs/.../best/best_program.py').read()); print('OK')"
```
If syntax is fine, check that the EVOLVE block does not shadow or delete `run_agentic` after the
fixed-infrastructure section.

### Scenario: All tasks timeout even with TBENCH_MAX_TASK_TIMEOUT=300

**Diagnosis**: Docker image builds are taking most of the 300s budget before the scaffold even starts.
Some TB2 tasks have Dockerfiles that take 3-5 minutes to build.

**Fix**: Pre-build all 89 Docker images before the run:
```bash
for d in /home/mertcemri/tb2_harbor_wrapped/*/; do
  name=$(basename $d)
  tag="skydiscover-harbor-${name}:latest"
  docker build -t "$tag" "$d/environment/" 2>/dev/null || echo "FAILED: $name"
done
```
The HarborEvaluator will find the cached image and skip the build step.

### Scenario: The best scaffold uses `two_phase` profile and scores drop after 30 iterations

**Diagnosis**: AdaEvolve has evolved into a two-phase configuration that happens to work on the
training set but is fragile on harder tasks.

**Fix**: Add `TBENCH_SEED_PROFILE=single_agent_native` to the run environment to prevent evolution
from using the two-phase path. Alternatively, set `SEED_NUM_AGENTS = 1` in the scaffold's
EVOLVE-BLOCK to hard-code single-agent.

### Scenario: `configure-git-webserver` always crashes with `run_agentic` error

**Diagnosis**: This task's test.sh uses a non-standard import path that the scaffold runner doesn't
match. The HarborEvaluator's `_extract_scaffold_runner_solution_path()` method should have fixed
this by reading `spec_from_file_location("solution", "/path")` from test.sh.

**Debug steps**:
```bash
# Check what path the evaluator infers for this task
python3 -c "
from skydiscover.evaluation.harbor_evaluator import HarborEvaluator
h = HarborEvaluator.__new__(HarborEvaluator)
h.task_dir = '/home/mertcemri/tb2_harbor_wrapped/configure-git-webserver'
print(h._extract_solution_path())
"
```

---

## Artifact Policy

### What to save from each phase

| Phase | Artifact | Location |
|---|---|---|
| Phase 0 | Smoke run stdout | `/tmp/smoke_fixed_seed.log` |
| Phase 1 | Crashprobe JSON | `/tmp/crashprobe_fixed_seed.json` |
| Phase 2 | Ablation JSON | `/tmp/ablation_tb2sample_fixed.json` |
| Phase 3 | Hardmini JSON | `/tmp/hardmini_fixed_seed.json` |
| Phase 4 | Full run output | `outputs/terminal_bench_adaevolve_v4/` |
| Phase 4 | Best program | `outputs/terminal_bench_adaevolve_v4/best/best_program.py` |
| Phase 4 | Best metadata | `outputs/terminal_bench_adaevolve_v4/best/best_program_info.json` |
| Phase 5 | Test eval JSON | `outputs/tb2_eval_v4_final/test_eval.json` |
| Phase 5 | Full TB2 parallel | `outputs/tb2_eval_v4_parallel/` |

### What to record in the next progress report

After Phase 4 + 5 complete, write `progress_reports/progress_04_XX.md` covering:
1. Best scaffold score and completion on train split (compare to v3: 0.8332 / 8/10)
2. Dev split generalization check (mid-run and final)
3. Full TB2 completion rate (compare to v3: 7/89 = 7.87%)
4. Which tasks were newly solved vs still failing
5. Which seed fixes had the most visible impact (look at iteration 1-5 trajectory)
6. Recommendation for Phase 6 or further tuning

---

## Model and Cost Policy

### Solver model (running the scaffold)

- **Default**: `gpt-4o-mini`. Cost-efficient, 128K context, good tool-calling.
- **Upgrade to `gpt-4o`**: Only if Phase 2 ablation shows ≥ 2 more tasks solved. The cost
  difference across 75 iterations × 10 tasks/iteration is substantial (~10× more expensive).
- **Do not use**: `gpt-5.2` as the solver. That model is for the mutator LLM. Using it as the
  solver is expensive and may cause context/rate issues on large tasks.

### Mutator model (AdaEvolve's LLM for generating scaffold mutations)

- **Default**: `gpt-5.2` (configured in config.yaml under `llm.models`). This is the strongest
  available mutator and produces the best diff-based mutations.
- **Do not downgrade**: `gpt-4o-mini` as the mutator produces lower-quality mutations and the
  evolutionary signal will be weaker.

### AI feedback model

- **Default**: `gpt-5` (configured in config.yaml). Fires every 5 iterations.
- This is fine. The feedback is short (a few paragraphs) and infrequent enough to not dominate cost.

---

## Rationale for Key Decisions

### Why `tb2_balanced` as the training distribution?

TB1-converted tasks (what v3 trained on) are systematically easier than TB2:
- TB1 tasks were written for a simpler CLI agent format
- Harbor conversion adds a test.sh wrapper but doesn't add genuine difficulty
- The 36-task TB1 train set is dominated by text/data manipulation and short Python programs
- Real TB2 tasks include: QEMU, OCaml/CompCert, Stan/R, CUDA, WebAssembly, custom memory allocators

Training on `tb2_balanced` forces the scaffold to evolve strategies that generalize to the
true task distribution. The 7.87% → target 17-22% improvement depends entirely on this.

### Why TRAIN_TASKS=10 (not 20 or all 64)?

With 10 tasks per evaluation and ~9 minutes per evaluation:
- 75 iterations × 9 min = ~11 hours wall time — acceptable for an overnight run
- With 20 tasks: ~22 hours — too long for a single overnight run
- With 64 tasks: ~88 hours — completely infeasible

10 tasks gives enough diversity signal for AdaEvolve's fitness function while keeping wall time
manageable. The variance in score from 10 tasks is higher than from 64, but AdaEvolve's UCB
selection and island structure smooth this out over many iterations.

### Why start from `initial_program.py` and not from v3 best?

v3 best overfit to TB1-style tasks. Its prompts, verification strategy, and acceptance logic
were all tuned implicitly toward shorter, easier tasks with simpler acceptance commands. Starting
fresh from the fixed seed means AdaEvolve starts from a clean slate that is already honest about
TB2's actual requirements.

### Why cascade threshold 0.12 instead of 0.30?

The seed on honest TB2 tasks scores approximately 0.05-0.10 (maybe 0-1 tasks out of 10 complete).
A threshold of 0.30 would reject the seed AND most early mutations, making it impossible for
AdaEvolve to build a fitness signal. At 0.12, any candidate that completes ≥ 1 TB2 task (score ~0.10
+ small bonuses) passes to stage3 and gets a real evaluation. The stage1 gate at 0.05 still filters
out completely broken scaffolds.

---

## Summary

Four mandatory seed fixes have been applied to `initial_program.py`:
1. `_discover_acceptance_hint()` now probes `/app/tests/test.sh` first — the correct TB2 path
2. `ACCEPTANCE_TOKENS` no longer includes shell idioms (`test -f`, `grep -q`, `diff`, `cmp`) that cause false completion
3. `HEAVY_COMMAND_TOKENS` now includes `apt-get` and `apt` for proper timeout allocation
4. `ACCEPTANCE_PLAYBOOK` starts with `bash /app/tests/test.sh`

`config.yaml` has been updated: `max_iterations: 75`, cascade stage2 threshold `0.12`.

**Run order:**
1. **Phase 0 (smoke)** — 5 min — confirm fixed seed works on bundled easy tasks
2. **Phase 1 (crashprobe)** — 20 min — diagnostic read on hard-task robustness (not a hard gate)
3. **Phase 2 (tb2sample ablation)** — 30 min — select solver model; confirm ≥1 task solved
4. **Phase 3 (hardmini gate)** — 60 min — formal go/no-go: need ≥10% completion on hard tasks
5. **Phase 4 (main AdaEvolve)** — ~11 hrs — 75-iteration search on `tb2_balanced`, TRAIN_TASKS=10, gpt-4o-mini solver, gpt-5.2 mutator, TBENCH_MAX_TASK_TIMEOUT=300
6. **Phase 5 (final eval)** — 30 min — full 89-task TB2 parallel eval on best evolved scaffold

**Target outcome:** ≥15/89 (17%) on full TB2, up from 7/89 (7.87%) in v3. The training distribution fix and acceptance command fixes together are the primary expected source of improvement.
