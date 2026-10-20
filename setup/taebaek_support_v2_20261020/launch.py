"""Local Taebaek support launch; resume only this reward revision."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / f'cyclo-matplotlib-{os.getuid()}'))
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch
from refinement import SOURCE, RUN_NAME, TOTAL_UPDATES, validate_inputs, validate_resume


def choose(num_envs, logroot=None):
    logroot = logroot or ROOT / 'logs/rsl_rl/k1_mimic'
    candidates = list(logroot.glob(f'*_{RUN_NAME}_*/model_*.pt'))
    candidates = [p for p in candidates if 'smoke' not in p.parent.name]
    candidates.sort(key=lambda p: (int(p.stem.split('_')[-1]), p.stat().st_mtime), reverse=True)
    for path in candidates:
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
        except (OSError, EOFError, RuntimeError) as exc:
            print('Skipping unreadable checkpoint:', path, repr(exc), flush=True)
            continue
        # A readable checkpoint with incompatible settings is an error, not a
        # reason to silently start another 10,000 updates from the source.
        completed = validate_resume(data, num_envs)
        if int(path.stem.split('_')[-1]) != completed - 1:
            raise ValueError(f'Checkpoint filename/iteration mismatch: {path}')
        return path, completed
    if candidates:
        raise RuntimeError('Existing Taebaek SupportV2 checkpoints are unreadable. Inspect them before restarting.')
    return SOURCE, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-gpu-sharing', action='store_true',
                        help='Explicitly allow an existing GPU process, e.g. a small viewer.')
    args = parser.parse_args()
    if args.gpu < 0 or args.num_envs < 4 or args.num_envs % 4:
        parser.error('gpu must be nonnegative; num-envs must be a positive multiple of 4.')
    if (HERE / "tracking_guard_halt.json").exists():
        raise RuntimeError("Tracking regression was detected; review tracking_guard_halt.json before restarting.")
    validate_inputs()
    checkpoint, completed = choose(args.num_envs)
    remaining = TOTAL_UPDATES - completed
    print(f'GPU={args.gpu}; num_envs={args.num_envs}; completed={completed}; remaining={remaining}; checkpoint={checkpoint}', flush=True)
    if args.dry_run or not remaining:
        return
    lock = os.fdopen(os.open(HERE / 'training.lock', os.O_RDONLY | os.O_CREAT, 0o644), 'r')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv,noheader'], text=True)
    busy = [line for line in apps.splitlines() if line.split(',')[0].strip() == uuid]
    if busy and not args.allow_gpu_sharing:
        raise RuntimeError('GPU is occupied. Close its viewer/training first, or explicitly use '
                           '--allow-gpu-sharing if enough memory is available.\n' + '\n'.join(busy))
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MUJOCO_GL='egl')
    os.environ.setdefault('WARP_CACHE_PATH', str(HERE / 'warp_cache'))
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    command = [sys.executable, '-u', str(HERE / 'train.py'), '--checkpoint', str(checkpoint),
        '--gpu-ids', f'[{args.gpu}]', '--env.scene.num-envs', str(args.num_envs),
        '--agent.max-iterations', str(remaining), '--agent.run-name', f'{RUN_NAME}_{tag}',
        '--agent.logger', 'tensorboard', '--agent.resume', 'False', '--agent.upload-model', 'False']
    os.set_inheritable(lock.fileno(), True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
