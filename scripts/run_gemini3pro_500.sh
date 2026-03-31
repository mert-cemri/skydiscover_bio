#!/usr/bin/env bash
# Run gemini-3-pro-preview experiments: 5 benchmarks x 2 strategies x 3 seeds = 30 runs
# All in parallel, 500 iterations, with nohup so they survive terminal close.
set -euo pipefail

MODEL="gemini-3-pro-preview"
ITERATIONS=500
OUTBASE="outputs/gemini3pro_v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Benchmarks ───────────────────────────────────────────────────────────
declare -A BENCH_INIT BENCH_EVAL BENCH_CONFIG
BENCH_INIT=(
  [circle_packing]="benchmarks/math/circle_packing/initial_program.py"
  [prism]="benchmarks/ADRS/prism/initial_program.py"
  [convex13]="benchmarks/math/heilbronn_convex/13/initial_program.py"
  [signal]="benchmarks/math/signal_processing/initial_program.py"
  [cloudcast]="benchmarks/ADRS/cloudcast/initial_program.py"
)
BENCH_EVAL=(
  [circle_packing]="benchmarks/math/circle_packing/evaluator.py"
  [prism]="benchmarks/ADRS/prism/evaluator/evaluator.py"
  [convex13]="benchmarks/math/heilbronn_convex/13/evaluator/evaluator.py"
  [signal]="benchmarks/math/signal_processing/evaluator/evaluator.py"
  [cloudcast]="benchmarks/ADRS/cloudcast/evaluator/evaluator.py"
)
BENCH_CONFIG=(
  [circle_packing]="benchmarks/math/circle_packing/config.yaml"
  [prism]="benchmarks/ADRS/prism/config.yaml"
  [convex13]="benchmarks/math/heilbronn_convex/13/config.yaml"
  [signal]="benchmarks/math/signal_processing/config.yaml"
  [cloudcast]="benchmarks/ADRS/cloudcast/config.yaml"
)

BENCHMARKS=(circle_packing prism convex13 signal cloudcast)
STRATEGIES=(adaevolve evox)
SEEDS=(1 2 3)

# ── Launch ───────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

COUNT=0
PIDS=()

for bench in "${BENCHMARKS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      outdir="${OUTBASE}/${bench}_${strategy}_seed${seed}"
      mkdir -p "$outdir"

      nohup uv run skydiscover-run \
        "${BENCH_INIT[$bench]}" \
        "${BENCH_EVAL[$bench]}" \
        -c "${BENCH_CONFIG[$bench]}" \
        -s "$strategy" \
        -m "$MODEL" \
        -i "$ITERATIONS" \
        -o "$outdir" \
        > "$outdir/run.log" 2>&1 &

      PIDS+=($!)
      COUNT=$((COUNT + 1))
      echo "[$COUNT] $bench / $strategy / seed$seed  (PID $!)"
    done
  done
done

echo ""
echo "Launched $COUNT runs at $OUTBASE"
mkdir -p "$OUTBASE"
echo "${PIDS[*]}" > "${OUTBASE}/pids_${TIMESTAMP}.txt"
echo "PIDs saved to ${OUTBASE}/pids_${TIMESTAMP}.txt"
echo ""
echo "Monitor with:"
echo "  for d in ${OUTBASE}/*/; do name=\$(basename \"\$d\"); ckpt=\$(ls -d \"\$d/checkpoints/checkpoint_\"* 2>/dev/null | sort -t_ -k2 -n | tail -1 | xargs basename 2>/dev/null); echo \"\$name  \$ckpt\"; done"
