"""Host-side, one-shot gated launcher. Invoked by the 30-minute heartbeat."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTAINER_ROOT = Path('/workspace/cyclo_mjlab')
RUN = '2026-10-19_11-17-37_taeguek07_arm_smooth_v1_finetune_20261019_111729'
RUN_NAME = 'taebaek_contact_corrected_scratch50000_after_t7_20261019'
NAME = 'cyclo_taebaek_after_t7_50000_20261019'
LOG_ROOT = ROOT / 'logs/rsl_rl/k1_mimic'
CHECKPOINT = LOG_ROOT / RUN / 'model_9999.pt'
STATE = HERE / 'launch_state.json'
MOTION = ROOT / 'source/assets/motions/K1_rev1/taebaek/k1_taebaek_speed100_contact_corrected_50hz.npz'
MOTION_SHA = '0c9f0b8e28acb6c7c7a9dcfee1a0670966d56032c4b514872f1c35b405657f9f'
REVISION = '25d007624b3c6960ba324d9ef95f855f14094a1f57ba87a54aea74b1cd9fdc14'

def run(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()

def emit(status, **kwargs):
    print(json.dumps({'status': status, **kwargs}, ensure_ascii=False, indent=2))

def container_state():
    p = subprocess.run(['docker', 'inspect', NAME], text=True, capture_output=True)
    if p.returncode == 0:
        return json.loads(p.stdout)[0]['State']
    if 'no such object' not in p.stderr.lower():
        raise RuntimeError(p.stderr)
    return None

def target_alive():
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = proc.read_bytes().split(b'\0')
            if b'taeguek07_arm_smooth_v1_finetune_20261019_111729' in args:
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return False

def validate_checkpoint(path):
    code = '''import torch,json,sys
p=sys.argv[1]; d=torch.load(p,map_location="cpu",weights_only=False)
assert d["iter"]==9999, d["iter"]
assert d["infos"]["env_state"]["common_step_counter"]==240000
assert d["infos"]["taeguek07_arm_smooth_v1"]["revision"]==sys.argv[2]
for key in ("actor_state_dict","critic_state_dict"):
 assert d[key] and all(torch.isfinite(v).all().item() for v in d[key].values())
assert d["optimizer_state_dict"]["state"]
print(json.dumps({"completed_updates":d["iter"]+1,"common_step_counter":240000}))
'''
    path_in_container = CONTAINER_ROOT / path.relative_to(ROOT)
    return json.loads(run(['docker', 'exec', 'cyclo_mjlab', 'python', '-c', code,
                           str(path_in_container), REVISION]))

def command():
    return ['docker', 'run', '-d', '--init', '--name', NAME,
            '--gpus', 'device=0', '--shm-size=16g', '--user', f'{os.getuid()}:{os.getgid()}',
            '--mount', f'type=bind,source={ROOT},target={CONTAINER_ROOT}',
            '--workdir', str(CONTAINER_ROOT),
            '--env', 'HOME=/tmp', '--env', 'MPLCONFIGDIR=/tmp/matplotlib',
            '--env', 'MUJOCO_GL=egl', '--env', 'OMP_NUM_THREADS=4', '--env', 'MKL_NUM_THREADS=4',
            '--env', f'PYTHONPATH={CONTAINER_ROOT}',
            '--env', f'WARP_CACHE_PATH={CONTAINER_ROOT}/setup/{HERE.name}/warp_cache',
            '--log-opt', 'max-size=50m', '--log-opt', 'max-file=4',
            '--entrypoint', 'python', 'cyclo-mjlab:latest',
            '-u', f'setup/{HERE.name}/train_taebaek.py', '--gpu-ids', '[0]',
            '--env.scene.num-envs', '4096', '--agent.max-iterations', '50000',
            '--agent.run-name', RUN_NAME, '--agent.logger', 'tensorboard',
            '--agent.resume', 'False', '--agent.upload-model', 'False']

def main(start):
    existing = container_state()
    if existing is not None or STATE.exists():
        emit('already_launched_or_attempted', container=existing,
             state=json.loads(STATE.read_text()) if STATE.exists() else None,
             runs=[str(p) for p in LOG_ROOT.glob('*' + RUN_NAME)])
        return
    if not CHECKPOINT.exists():
        paths=list(CHECKPOINT.parent.glob('model_*.pt'))
        latest=max(paths, key=lambda p:int(p.stem.split('_')[-1]), default=None)
        emit('waiting_for_t7' if target_alive() else 't7_stopped_before_completion',
             target_run=RUN, latest_checkpoint=str(latest), planned_command=command())
        return
    if target_alive():
        emit('waiting_for_t7_process_exit', target_run=RUN)
        return
    s=CHECKPOINT.stat()
    if time.time()-s.st_mtime < 60:
        emit('waiting_for_checkpoint_to_settle')
        return
    completion=validate_checkpoint(CHECKPOINT)
    assert (s.st_size,s.st_mtime_ns)==(CHECKPOINT.stat().st_size,CHECKPOINT.stat().st_mtime_ns)
    gpu=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name','--format=csv,noheader'])
    if gpu:
        emit('waiting_for_gpu', processes=gpu, completion=completion)
        return
    assert hashlib.sha256(MOTION.read_bytes()).hexdigest()==MOTION_SHA, 'Taebaek motion changed'
    assert shutil.disk_usage(ROOT).free > 10*1024**3, 'Less than 10 GiB free disk'
    assert not list(LOG_ROOT.glob('*'+RUN_NAME)), 'Taebaek run already exists'
    if not start:
        emit('ready', completion=completion, planned_command=command())
        return
    (HERE/'warp_cache').mkdir(exist_ok=True)
    state={'status':'launch_attempted','created_unix':time.time(),'target_t7':RUN,
           'completion':completion,'motion_sha256':MOTION_SHA,'command':command()}
    with STATE.open('x') as f:
        json.dump(state,f,indent=2); f.flush(); os.fsync(f.fileno())
    cid=run(command())
    state.update(status='container_started',container_id=cid)
    tmp=STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(state,indent=2)); tmp.replace(STATE)
    emit('container_started',container_id=cid,run_name=RUN_NAME)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start-if-ready',action='store_true')
    args=parser.parse_args()
    with (HERE/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args.start_if_ready)
