"""User-launched fine-tuning. Re-run to resume this task's newest checkpoint."""
import argparse
import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

# Configure this before refinement imports Matplotlib through the training stack.
# Separate by UID so root checks cannot create cyclo's cache with root ownership.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / f'cyclo-matplotlib-{os.getuid()}'))

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from refinement import MARKER, SOURCE, SOURCE_SHA, TOTAL_UPDATES, revision, validate_motion


def choose():
    source = SOURCE
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError('Complete 7 checkpoint fingerprint mismatch.')
    paths = list((ROOT / 'logs/rsl_rl/k1_mimic').glob('*_taeguek07_arm_smooth_v1_finetune_*/model_*.pt'))
    paths.sort(key=lambda p: (int(p.stem.split('_')[-1]), p.stat().st_mtime), reverse=True)
    for path in paths:
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            meta = (data.get('infos') or {}).get(MARKER)
            if not meta or meta.get('sha256') != SOURCE_SHA or meta.get('revision') != revision():
                continue
            completed = int(data['iter']) + 1
            assert int(path.stem.split('_')[-1]) == int(data['iter'])
            assert 0 < completed <= TOTAL_UPDATES
            assert data['infos']['env_state']['common_step_counter'] == completed * 24
            assert 'optimizer_state_dict' in data
            assert all(torch.isfinite(v).all() for key in ('actor_state_dict', 'critic_state_dict') for v in data[key].values())
        except Exception as exc:
            print('Skipping incomplete checkpoint:', path, repr(exc), flush=True)
            continue
        return path, completed
    return source, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int, choices=(0,))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    validate_motion()
    checkpoint, completed = choose()
    remaining = max(0, TOTAL_UPDATES - completed)
    print(f'GPU={args.gpu}; completed={completed}; remaining={remaining}; checkpoint={checkpoint}', flush=True)
    if args.dry_run or not remaining:
        return
    # flock does not require a writable descriptor on this local Linux mount.
    # Read-only opening also accepts a lock previously created by another UID.
    lock = os.fdopen(os.open(HERE / 'training.lock', os.O_RDONLY | os.O_CREAT, 0o644), 'r')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == uuid for line in apps.splitlines()):
        raise RuntimeError(f'GPU {args.gpu} is occupied. Choose an idle GPU.\n{apps}')
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MUJOCO_GL='egl')
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    command = [sys.executable, '-u', str(HERE / 'train.py'), '--checkpoint', str(checkpoint),
        '--gpu-ids', f'[{args.gpu}]', '--env.scene.num-envs', '4096',
        '--agent.max-iterations', str(remaining),
        '--agent.run-name', f'taeguek07_arm_smooth_v1_finetune_{tag}',
        '--agent.logger', 'tensorboard', '--agent.resume', 'False', '--agent.upload-model', 'False']
    # Inherit the lock across exec so two launchers cannot train this task twice.
    os.set_inheritable(lock.fileno(), True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
