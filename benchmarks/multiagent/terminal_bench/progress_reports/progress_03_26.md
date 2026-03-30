# Terminal Bench Progress Report
Date: 2026-03-26

## Executive Summary

Progress so far is real but uneven.

The main positive result is that the search run `outputs/terminal_bench_adaevolve_v3` found a substantially stronger scaffold than the seed on the internal optimization benchmark. The saved best program reached `combined_score = 0.8332` and completed `8/10` tasks on the search-time evaluation set.

The main negative result is that this improvement does not yet generalize well to the harder, broader TB2-style regime. The saved full-TB2 parallel evaluation of the current v3 best scaffold solved only `7/89` tasks, for a completion rate of `7.87%`.

The clearest current diagnosis is:

- Search can improve the scaffold on the curated optimization set.
- Robustness on hard tasks is still poor.
- Generalization to full TB2 is still poor.
- `single_agent_native` is clearly better than `two_phase` in the saved artifacts.
- The remaining blockers are mostly hard-task execution, dependency/setup handling, timeout behavior, verification discipline, and some probable evaluation-wrapper edge cases.

## Artifact Inventory Reviewed

This report is based on the following saved artifacts:

- `outputs/terminal_bench_adaevolve_v3/best/best_program_info.json`
- `outputs/terminal_bench_adaevolve_v3/checkpoints/checkpoint_50/adaevolve_metadata.json`
- `outputs/tb2_eval_v3_parallel/summary.json`
- `outputs/tb2_eval_v3_parallel/run.log`
- `outputs/tb2_eval_v3_parallel/per_task/*/result.json`
- `outputs/terminal_bench_adaevolve/hard_mini_suite.json`
- `outputs/terminal_bench_adaevolve/ablation_tb2_matrix.json`
- `outputs/terminal_bench_adaevolve/ablation_smoke.json`
- `outputs/terminal_bench_adaevolve/ablation_smoke_single.json`
- `outputs/terminal_bench_adaevolve/ablation_tb2sample_gpt-4o-mini_single_agent_native_20260324T162219Z.json`
- `outputs/terminal_bench_adaevolve/tb2_crashprobe_gpt-4o-mini_single_agent_native_t60_20260324T082255Z.json`
- `outputs/terminal_bench_adaevolve/tb2_crashprobe_gpt-4o-mini_single_agent_native_t60_20260325T000242Z.json`
- `outputs/terminal_bench_adaevolve/tb2_crashprobe_gpt-4o-mini_single_agent_native_t60_20260326T084244Z.json`

## Highest-Level Status

### 1. Search quality improved materially

The strongest saved result is the v3 best program:

- File: `outputs/terminal_bench_adaevolve_v3/best/best_program_info.json`
- Best generation: `3`
- Best iteration: `27`
- Combined score: `0.8331666666666667`
- Completion rate: `0.8`
- Tasks completed: `8/10`
- Done-signal count: `10/10`
- Average steps per task: `10.1`
- Average latency per task: `17.1s`
- Agents used: `4`

This means the optimization loop was not wasted. It found a scaffold that is substantially better than the seed on the internal train-time benchmark.

### 2. Full TB2 performance is still weak

The full parallel TB2 evaluation summary is:

- File: `outputs/tb2_eval_v3_parallel/summary.json`
- Program evaluated: `outputs/terminal_bench_adaevolve_v3/best/best_program.py`
- Total TB2 tasks: `89`
- Tasks completed: `7`
- Completion rate: `0.0787`
- Mean score: `0.0787`
- Total wall time: `702.2s`

So the best evolved scaffold currently solves only `7/89` TB2 tasks.

### 3. Hard-task robustness is still poor

Three saved artifact families all show the same pattern:

- `hard_mini_suite.json`: current scaffold fails hard tasks badly
- `tb2_crashprobe_*.json`: repeated timeout failures on selected hard tasks
- Full TB2 evaluation: almost all complex tasks still fail

This is the main bottleneck now.

## Search Run Status: `terminal_bench_adaevolve_v3`

### Best result

The best program found so far achieved:

- `combined_score = 0.8332`
- `n_completed = 8`
- `n_attempted = 10`
- `n_done_signal = 10`
- `num_agents = 4`
- `total_steps = 101`
- `avg_steps_per_task = 10.1`

Interpretation:

- The scaffold almost always believes it is finished.
- On the internal 10-task benchmark, this was usually correct enough to score well.
- Efficiency was reasonable: about 10 steps per task on average.
- The best architecture selected by search uses 4 agents.

### Search dynamics

The checkpoint metadata shows:

- Total iterations: `50`
- Global best score: `0.8331666666666667`
- Number of islands: `2`
- Both islands labeled `balanced`
- Adaptive/UCB search enabled
- Improvement counts per adapter state: `4` and `5`

Interpretation:

- The run was long enough to make multiple genuine improvements.
- The run did not plateau immediately.
- The search infrastructure appears healthy enough to discover improved scaffolds.

### What this means

The optimizer can improve the scaffold inside the chosen training distribution. The problem is not "search is broken". The problem is that the discovered solution is still heavily overfit or underpowered for real TB2 breadth.

## Smoke and Small Ablation Status

### Smoke result on simple task

`ablation_smoke.json` and `ablation_smoke_single.json` show that the simple smoke task is solvable by the single-agent-native scaffold.

Single-agent-native smoke result:

- `combined_score` about `1.04`
- `completion_rate = 1.0`
- `n_completed = 1/1`
- `done_signal = 1`
- Verified success

Interpretation:

- The scaffold can still complete easy structured tasks end-to-end.
- The base runtime is not universally broken.

### Two-phase seed status

The two-phase configuration looks substantially worse in the saved artifacts.

Observed behavior:

- In smoke evaluation it crashes with an OpenAI API 400
- In broader ablation matrix runs it is usually at `0/5`
- On hard-mini it times out on all `6/6` tasks

Interpretation:

- `two_phase` is not competitive right now.
- It is likely either functionally broken under the current prompt/runtime assumptions or too slow/fragile to survive the benchmark.

Conclusion:

- The working branch of progress is `single_agent_native`, not `two_phase`.

## TB2 Sample / Harbor Sample Status

The saved TB2-sample ablation artifact for `gpt-4o-mini + single_agent_native` is poor.

Train split result:

- `n_attempted = 2`
- `n_completed = 0`
- `n_no_progress = 2`
- `n_timeout = 0`
- `n_scaffold_crash = 0`

Dev split result:

- `n_attempted = 2`
- `n_completed = 0`
- `n_no_progress = 2`

Interpretation:

- The scaffold is not just failing with crashes or timeouts.
- It is often entering a state the evaluator classifies as `no_progress`.
- This strongly suggests weaknesses in action selection, verification sequencing, or acceptance-command handling, not just raw runtime errors.

This matches the code-level audit concerns in `initial_program.py`:

- dependency installation strategy is weak
- verification can be too shallow
- acceptance command handling is brittle
- there is no strong no-progress loop breaker

## Hard Mini Suite Status

The saved hard-mini suite result is one of the clearest warnings.

For `gpt-4o-mini + single_agent_native`:

- `n_attempted = 6`
- `n_completed = 0`
- `completion_rate = 0.0`
- `n_scaffold_crash = 1`
- `n_task_failure = 5`
- `n_timeout = 0`

The six hard tasks shown in the artifact are:

- `sam-cell-seg`
- `compile-compcert`
- `reshard-c4-data`
- `mcmc-sampling-stan`
- `llm-inference-batching-scheduler`
- `path-tracing`

Observed failure patterns:

- `sam-cell-seg`: large dependency setup and repeated failed verification
- `compile-compcert`: multi-step build/setup attempt but no successful finish
- `reshard-c4-data`: writes code, repeatedly fails pytest acceptance
- `mcmc-sampling-stan`: repeated R/rstan installation failures
- `llm-inference-batching-scheduler`: scaffold crash
- `path-tracing`: some progress, but acceptance still fails

Interpretation:

- The scaffold can often take actions and write plausible code.
- It still fails to close the loop on hard environments.
- The hard-suite failures are consistent with weak dependency handling, weak install retry logic, and insufficient time budget or progress management.

## Crashprobe Status

The crashprobe artifacts from `2026-03-24`, `2026-03-25`, and `2026-03-26` all show effectively the same result:

- `n_attempted = 3`
- `n_completed = 0`
- `n_timeout = 3`
- `completion_rate = 0.0`

Affected tasks:

- `sam-cell-seg`
- `llm-inference-batching-scheduler`
- `reshard-c4-data`

Interpretation:

- There has been no visible improvement on the crashprobe set across those saved checkpoints.
- This is important because it suggests the current scaffold is still highly vulnerable to long-running or setup-heavy tasks.
- It also matches the code issue where the PTY shell clamps command timeout to 60s even though higher timeouts are configured elsewhere.

This is one of the strongest indicators that hard-task robustness has not improved yet.

## Full TB2 Evaluation Status

### Aggregate result

The saved full-TB2 run finished successfully as an evaluation job and produced a complete summary:

- Total tasks: `89`
- Completed: `7`
- Completion rate: `7.87%`
- Mean score: `7.87%`
- Evaluation completed without aggregate runner errors

This means the evaluation infrastructure worked well enough to produce a trustworthy broad result.

### Passed tasks

The seven passed tasks are:

- `constraints-scheduling`
- `fix-git`
- `git-leak-recovery`
- `log-summary-date-ranges`
- `regex-log`
- `sqlite-with-gcov`
- `vulnerable-secret`

These tasks cluster roughly into:

- text/log/data processing
- simpler shell/repo repair
- constrained procedural tasks
- some medium build/setup tasks with shorter, clearer solution paths

### Failed tasks

The remaining `82/89` tasks failed.

Broadly failed categories include:

- heavy compilation/build tasks
- ML/data-science tasks
- cryptography tasks
- QEMU/VM tasks
- advanced systems tasks
- larger search/optimization tasks
- specialized language/runtime tasks

Examples of especially long failures from `run.log`:

- `query-optimize` at `575s`
- `qemu-startup` at `417s`
- `custom-memory-heap-crash` at `446s`
- `crack-7z-hash` at `396s`

Interpretation:

- Many tasks consume substantial wall-clock time before failing.
- The failures are not all immediate crashes; a large fraction are expensive failed attempts.
- This means the scaffold is "trying" many hard tasks, but not succeeding.

## Important Per-Task Diagnostics

### Example of true success

`constraints-scheduling` succeeded cleanly:

- `completed = true`
- `score = 1.0`
- `done_signal = 1`
- `num_agents = 4`
- `total_steps = 11`

This shows the best scaffold is not a fluke or broken artifact. It can execute a full multi-agent solve/verify cycle successfully on real TB2 tasks.

### Example of success on a medium systems/build task

`sqlite-with-gcov` also succeeded:

- `completed = true`
- `score = 1.0`
- `done_signal = 1`
- `num_agents = 4`
- `total_steps = 16`

This is encouraging because it shows the scaffold can handle some real build/setup tasks, not just text editing.

### Example of false-positive completion behavior

`query-optimize` is a very informative failure:

- `completed = false`
- `score = 0.0`
- `done_signal = 1`
- `total_steps = 12`
- `num_agents = 4`

Interpretation:

- The scaffold believed it had finished.
- The evaluator did not accept the output.
- This is exactly the kind of verifier/acceptance mismatch that the `initial_program.py` audit warned about.

### Example of likely infrastructure/wrapper edge case

`configure-git-webserver` contains:

- `AttributeError: module 'solution' has no attribute 'run_agentic'`

Interpretation:

- At least some TB2 failures may include evaluation-wrapper or packaging issues, not only scaffold-quality issues.
- This does not explain the entire poor TB2 result, because many other failures clearly come from genuine task failure.
- It is still worth treating as a real infrastructure caveat when interpreting the `7/89` headline.

## Comparison Table

| Evaluation | Program / Setting | Result |
|---|---|---|
| Smoke task | `single_agent_native` | Pass |
| Smoke task | `two_phase` | Fail / API crash |
| TB2 sample train/dev | `single_agent_native` | `0/2`, `0/2`, both dominated by `no_progress` |
| Hard mini suite | `single_agent_native` | `0/6` |
| Crashprobe | `single_agent_native` | `0/3`, repeated over multiple dates |
| Search-time v3 best evaluation | `terminal_bench_adaevolve_v3` best | `8/10`, score `0.8332` |
| Full TB2 parallel eval | v3 best program | `7/89`, `7.87%` |

## What Has Improved

The following are genuine wins:

- The search process can discover improved scaffolds.
- A best scaffold was found with strong internal benchmark performance.
- The discovered scaffold can solve some real TB2 tasks.
- The full-TB2 parallel evaluation infrastructure exists and produced a complete run.
- The optimization setup has moved beyond smoke-only validation.

This is meaningful progress. The project is not stalled at the infrastructure-only stage anymore.

## What Is Still Broken or Weak

The following remain clear problems:

- Hard-task robustness is poor.
- Dependency installation strategy is weak.
- Long-running commands still appear bottlenecked by timeout behavior.
- The scaffold sometimes emits `done_signal = 1` without true task completion.
- The acceptance-command logic is likely too brittle.
- There is not enough no-progress loop detection or recovery.
- `two_phase` is not viable in the current form.
- Some evaluation artifacts suggest wrapper or entrypoint inconsistencies on specific tasks.

## Most Likely Current Failure Modes

Based on the saved results and the code audit, the dominant failure modes are:

### 1. Dependency/setup failure on hard tasks

Observed in tasks like:

- `sam-cell-seg`
- `mcmc-sampling-stan`
- `path-tracing`
- many full-TB2 tasks involving external toolchains or larger runtimes

### 2. Timeout pressure on hard tasks

Observed in:

- crashprobe
- long-running full-TB2 tasks
- tasks that should benefit from longer command budgets

Likely connected to the `60s` PTY clamp inside `InteractiveShellSession.run_command`.

### 3. Weak verification / acceptance coupling

Observed in:

- `query-optimize` and similar cases where the scaffold claims completion but scores 0
- sample tasks labeled `no_progress`
- hard-suite tasks that repeatedly run weak checks without landing a decisive acceptance success

### 4. No-progress loops

Observed explicitly in TB2-sample artifacts:

- repeated `n_no_progress`
- no crashes, no timeouts, but also no real completion

### 5. Some infrastructure caveats still remain

Observed in:

- `configure-git-webserver` with missing `run_agentic`
- occasional solution-path warnings
- wrapper-level caveats that may affect a minority of results

## Overall Assessment

The project is in a classic "promising internal progress, poor held-out generalization" state.

The best current statement of progress is:

- Internal optimization benchmark: good progress
- Simple smoke tasks: working
- Hard-mini and crash probes: poor
- Full TB2: poor but nonzero capability

If reduced to one sentence:

The search has discovered a scaffold that is clearly better than the seed on the curated train-time evaluation, but it is still far from robust enough for full TB2 deployment.

## Recommended Immediate Focus

Based on the saved results so far, the highest-value next work is:

- Fix timeout handling so heavy commands can actually use the configured extended timeout.
- Strengthen dependency installation and transient retry behavior.
- Tighten verification so only meaningful acceptance evidence can satisfy completion.
- Broaden and normalize acceptance-command discovery.
- Add explicit no-progress / repetition detection.
- Keep focusing on `single_agent_native`; treat `two_phase` as low priority until the current branch is stable.
- Re-run crashprobe and hard-mini after those fixes before trusting another full-TB2 evaluation.

## Final Status Snapshot

As of 2026-03-26:

- Best saved scaffold: `outputs/terminal_bench_adaevolve_v3/best/best_program.py`
- Best search score: `0.8332`
- Best search completion: `8/10`
- Full TB2 completion: `7/89`
- Hard-mini completion: `0/6`
- Crashprobe completion: `0/3`
- Leading branch: `single_agent_native`
- Main concern: severe generalization gap plus hard-task timeout/setup fragility
