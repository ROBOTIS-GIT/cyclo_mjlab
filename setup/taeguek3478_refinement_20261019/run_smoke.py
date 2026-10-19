from pathlib import Path
import os,json,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.chdir(ROOT)
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MUJOCO_GL='egl')
for chapter,s in json.loads((HERE/'sources.json').read_text()).items():
    command=[sys.executable,'-u',str(HERE/'train_targeted.py'),chapter,
        '--env.scene.num-envs','32','--agent.max-iterations','2',
        '--agent.run-name',f'taeguek{int(chapter):02d}_targeted_v3_smoke_20261019',
        '--agent.logger','tensorboard','--agent.resume','False','--agent.upload-model','False',
        '--checkpoint',str(Path(s['source_run'])/s['checkpoint']),'--gpu-ids','[0]']
    print('SMOKE_START',chapter,flush=True)
    with (HERE/f'smoke_t{chapter}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('SMOKE_COMPLETE',chapter,flush=True)
