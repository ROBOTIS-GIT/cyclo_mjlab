"""Run on the remote host. Detached container exec; idempotent launch record."""
from pathlib import Path
import subprocess,json,shlex,datetime,os
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RECORD=HERE/'launch.json'
if RECORD.exists():
    print(RECORD.read_text());raise SystemExit('Already launched; inspect existing run, do not duplicate')
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
if gpu:raise RuntimeError('GPU 0 has a compute job: '+gpu)
run='taeguek08_arm_refined_support_scratch_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
base=Path('/workspace/cyclo_mjlab')
log=base/'setup'/HERE.name/(run+'.log')
command=['python','-u',f'setup/{HERE.name}/train_scratch.py','--gpu-ids','[0]',
         '--env.scene.num-envs','4096','--agent.max-iterations','30000','--agent.run-name',run,
         '--agent.logger','tensorboard','--agent.upload-model','False']
args=['docker','exec','-d','--user',f'{os.getuid()}:{os.getgid()}','-w',str(base),
      '-e','HOME=/tmp','-e','MPLCONFIGDIR=/tmp/taeguek08_mpl','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
      '-e','MUJOCO_GL=egl','-e',f'WARP_CACHE_PATH={base}/setup/{HERE.name}/warp_cache',
      'cyclo_mjlab','sh','-c','exec '+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1']
state={'run_name':run,'host':'robotis@192.168.60.177','log_file':str(HERE/log.name),'container_log_file':str(log),
       'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':args,'status':'launch_attempted'}
with RECORD.open('x') as f:json.dump(state,f,indent=2)
subprocess.run(args,check=True)
state['status']='detached_exec_started';RECORD.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
