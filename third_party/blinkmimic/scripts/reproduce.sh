#!/bin/bash
# Reproduce the main result: 3 seeds of the released recipe, every checkpoint evaluated on all 40 clips with 4
# replicas (the simulator is not bitwise reproducible, so a borderline clip can flip between replicas), then the time
# to target. Run on an otherwise idle GPU: the times are training wall-clock.
# Checkpoints are evaluated in the layout of the reported runs: iterations 100-900 together (9 per batch) and
# 1000-3500 together (26 per batch); the two result files are merged into <run>/eval_r4.csv.
# Usage: bash scripts/reproduce.sh [out_dir]   (default logs/reproduce; must not exist). Python: $PYTHON.
set -eu -o pipefail
cd "$(dirname "$0")/.."
OUT=${1:-logs/reproduce}
PY=${PYTHON:-uv run python}
[ -e "$OUT" ] && { echo "$OUT exists; choose a new out_dir" >&2; exit 1; }
for s in 0 1 2; do
  bash scripts/train.sh "$OUT/s$s" --seed "$s" > /dev/null
  test -f "$OUT/s$s/ckpt_003500.pt" || { echo "training of seed $s did not finish" >&2; exit 1; }
done
for s in 0 1 2; do
  R=$(cd "$OUT/s$s" && pwd)
  for part in early main; do
    D=$R/eval_$part; mkdir -p "$D"; cp "$R/args.json" "$D/"
    for c in "$R"/ckpt_*.pt; do
      i=$(basename "$c" .pt); i=$((10#${i#ckpt_}))
      if { [ $part = early ] && [ $i -lt 1000 ]; } || { [ $part = main ] && [ $i -ge 1000 ]; }; then ln -s "$c" "$D/"; fi
    done
    G=$([ $part = early ] && echo 9 || echo 26)
    PYTHONPATH=src $PY -u tools/eval_batch.py --run_dir "$D" --group $G --reps 4
  done
  { head -n 1 "$R/eval_early/eval_r4.csv"; tail -q -n +2 "$R/eval_early/eval_r4.csv" "$R/eval_main/eval_r4.csv"; } > "$R/eval_r4.csv"
done
$PY tools/time_to_target.py "$OUT"/s0 "$OUT"/s1 "$OUT"/s2
