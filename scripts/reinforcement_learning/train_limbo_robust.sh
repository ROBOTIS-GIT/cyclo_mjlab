#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python - <<'PY'
from pathlib import Path
import sys
running=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    if any(x.endswith((b'train.py',b'train_limbo.py')) for x in args) and any(b'Cyclo-Mimic-K1-Rev1-Limbo' in x for x in args):running.append(p.name)
if running:raise SystemExit('Limbo training already running; no duplicate started: '+str(running))
PY
exec python scripts/reinforcement_learning/train_limbo.py \
 Cyclo-Mimic-K1-Rev1-Limbo-1p2m-RobustObs124 \
 --warm-start "$PROJECT_ROOT/source/assets/checkpoints/limbo/robust_warm_start.pt" \
 --env.scene.num-envs 4096 --agent.max-iterations 20000 \
 --agent.run-name bar120_robust_v6 "$@"
