"""Managed local-container launcher; nothing starts until this is invoked."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
def writable_cache(variable, label):
    fallback=Path(tempfile.gettempdir())/f't8-contact-precision-v6-{label}-{os.getuid()}'
    preferred=Path(os.environ.get(variable,str(fallback)))
    for candidate in dict.fromkeys((preferred,fallback)):
        try:
            candidate.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryFile(dir=candidate):pass
        except OSError:
            continue
        os.environ[variable]=str(candidate)
        return
    raise PermissionError(f'No writable cache for {variable}')


writable_cache('MPLCONFIGDIR','mpl')
writable_cache('WARP_CACHE_PATH','warp')
os.environ.setdefault('OMP_NUM_THREADS','2')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('gpu',nargs='?',type=int,default=0)
    p.add_argument('--num-envs',type=int,default=None)
    p.add_argument('--resume',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--smoke',action='store_true',help='8 updates, defaults to 64 envs; plumbing only')
    a=p.parse_args()
    os.chdir(ROOT)
    import torch
    import source.tasks
    from config import (validate_inputs,revision,env_cfg,agent_cfg,ContactRunner,
                        completed_updates,TASK,TOTAL,MARKER,SOURCE,MOTION)
    from mjlab.tasks.registry import register_mjlab_task
    from scripts.reinforcement_learning.train import TrainConfig,launch_training
    manifest=validate_inputs()
    num_envs=a.num_envs if a.num_envs is not None else (64 if a.smoke else 4096)
    if num_envs<1 or a.gpu<0: raise ValueError('Invalid environment count/GPU')
    done=0
    if a.resume:
        a.resume=a.resume.resolve()
        done=completed_updates(torch.load(a.resume,map_location='cpu',weights_only=False),num_envs,a.smoke)
    remaining=(8 if a.smoke else TOTAL-done)
    if remaining<=0: raise ValueError('Training is already complete')
    plan=dict(task=TASK,gpu=a.gpu,num_envs=num_envs,initial_state='resume' if a.resume else 'warmstart',
              checkpoint=str(a.resume or SOURCE),completed_updates=done,
              remaining_updates=remaining,total_updates=TOTAL,smoke=a.smoke,
              revision=revision(),motion=str(MOTION),
              input_status=manifest['status'],automatic_sim2sim=False)
    print(json.dumps(plan,indent=2),flush=True)
    if a.dry_run:return
    # Writable outside setup even when bind-mounted files were created by another UID.
    key=hashlib.sha256(str(HERE).encode()).hexdigest()[:16]
    lock=Path(tempfile.gettempdir())/f't8-contact-{key}-{os.getuid()}.lock'
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not a.resume and not a.smoke:
            existing=[r for r in (ROOT/'logs/rsl_rl/k1_mimic').glob(f'*{MARKER}_finetune*')
                      if 'smoke' not in r.name and any(r.glob('model_*.pt'))]
            if existing:raise RuntimeError('Existing run found; use --resume with its checkpoint: '+str(existing[-1]))
        import subprocess
        usage=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if usage: raise RuntimeError('GPU already has compute processes: '+usage)
        cfg=env_cfg();cfg.scene.num_envs=num_envs
        agent=agent_cfg();agent.max_iterations=remaining
        if a.smoke:agent.run_name=MARKER+'_smoke';agent.save_interval=4
        register_mjlab_task(task_id=TASK,env_cfg=cfg,play_env_cfg=env_cfg(True),
                           rl_cfg=agent,runner_cls=ContactRunner)
        launch_training(TASK,TrainConfig(env=cfg,agent=agent,gpu_ids=[a.gpu],
                        checkpoint=str(a.resume or SOURCE)))


if __name__=='__main__':main()
