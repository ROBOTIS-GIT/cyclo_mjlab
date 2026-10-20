"""GPU0: verified selected weights -> isolated candidate training -> CPU exports.

Exports are unreviewed candidates. Never replace the installed sim2sim bundle.
"""
from pathlib import Path
import fcntl,json,os,subprocess,sys,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
lock=(HERE/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=(HERE/'driver.started').open('x');started.write(datetime.datetime.now(datetime.timezone.utc).isoformat());started.close()
(HERE/'driver.pid').write_text(str(os.getpid()));os.chdir(ROOT)
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MUJOCO_GL='egl')
status=1
try:
    subprocess.run([sys.executable,str(HERE/'validate_launch.py')],check=True)
    for c,s in json.loads((HERE/'sources.json').read_text()).items():
        chapter=int(c);run_name=f'taeguek{chapter:02d}_targeted_v3_finetune_20261019'
        print(datetime.datetime.now().isoformat(),'START',chapter,flush=True)
        command=[sys.executable,'-u',str(HERE/'train_targeted.py'),c,
            '--env.scene.num-envs','4096','--agent.max-iterations','10000',
            '--agent.run-name',run_name,'--agent.logger','tensorboard',
            '--agent.resume','False','--agent.upload-model','False',
            '--checkpoint',str(Path(s['source_run'])/s['checkpoint']),'--gpu-ids','[0]']
        with (HERE/f'train_t{c}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        runs=list((ROOT/'logs/rsl_rl/k1_mimic').glob('*_'+run_name));assert len(runs)==1,runs
        run=runs[0];assert (run/'model_9999.pt').is_file()
        output=HERE/'exports'/f't{c}_targeted_v3_9999';output.parent.mkdir(exist_ok=True)
        subprocess.run([sys.executable,str(HERE/'export_snapshot.py'),str(run),str(output),'model_9999.pt'],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
        (HERE/f'chapter_{chapter:02d}.complete.json').write_text(json.dumps(dict(chapter=chapter,
            run=str(run),export=str(output),quality_status='awaiting_full_sim2sim_evaluation'),indent=2))
        print(datetime.datetime.now().isoformat(),'COMPLETE',chapter,flush=True)
    status=0
finally:
    (HERE/'driver.exit').write_text(str(status))
