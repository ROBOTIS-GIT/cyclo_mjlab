"""User-launched GPU0:3->4 / GPU1:7->8. Does not stop other processes."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCES=json.loads((HERE/'sources.json').read_text());TOTAL=10000

def choose(chapter):
    src=SOURCES[str(chapter)];original=Path(src['source_run'])/src['checkpoint']
    assert hashlib.sha256(original.read_bytes()).hexdigest()==src['checkpoint_sha256']
    candidates=list((ROOT/'logs/rsl_rl/k1_mimic').glob(f'*_taeguek{chapter:02d}_targeted_v3_finetune_*/model_*.pt'))
    candidates.sort(key=lambda p:(int(p.stem.split('_')[-1]),p.stat().st_mtime),reverse=True)
    for path in candidates:
        try:
            state=torch.load(path,map_location='cpu',weights_only=False)
        except Exception as e:
            print('SKIP_INCOMPLETE_CHECKPOINT',path,type(e).__name__,flush=True);continue
        meta=(state.get('infos') or {}).get('targeted_3478')
        if not meta or meta.get('sha256')!=src['checkpoint_sha256']:continue
        completed=int(state['iter'])+1
        assert state['infos']['env_state']['common_step_counter']==completed*24,path
        assert 'optimizer_state_dict' in state and completed<=TOTAL,path
        assert all(torch.isfinite(v).all() for v in state['actor_state_dict'].values()),path
        return path,completed
    return original,0

def idle(gpu):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):
        raise RuntimeError(f'GPU {gpu} is occupied. Run manual_stop.py --apply first.\n{apps}')

def run_logged(command,log,env=None):
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
        bufsize=1,start_new_session=True,env=env)
    try:
        with log.open('w') as out:
            for line in child.stdout:
                print(line,end='',flush=True);out.write(line);out.flush()
        code=child.wait()
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
        raise
    if code:raise subprocess.CalledProcessError(code,command)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('gpu',type=int,choices=[0,1]);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();chapters=(3,4) if args.gpu==0 else (7,8)
    if args.dry_run:
        for c in chapters:
            path,n=choose(c);print(json.dumps(dict(gpu=args.gpu,chapter=c,checkpoint=str(path),completed=n,remaining=TOTAL-n)))
        return
    lock=(HERE/f'manual_gpu{args.gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    idle(args.gpu);os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MUJOCO_GL='egl')
    subprocess.run([sys.executable,str(HERE/'validate_launch.py')],check=True)
    tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
    status=HERE/f'manual_gpu{args.gpu}_{tag}.status.json'
    state=dict(gpu=args.gpu,chapters=chapters,pid=os.getpid(),status='running',completed=[])
    status.write_text(json.dumps(state,indent=2))
    try:
        for c in chapters:
            checkpoint,done=choose(c);remaining=TOTAL-done
            print(f'CHAPTER {c}: completed={done}, remaining={remaining}, checkpoint={checkpoint}',flush=True)
            if remaining:
                idle(args.gpu)
                name=f'taeguek{c:02d}_targeted_v3_finetune_manual_gpu{args.gpu}_{tag}'
                command=[sys.executable,'-u',str(HERE/'train_manual.py'),str(c),
                    '--env.scene.num-envs','4096','--agent.max-iterations',str(remaining),
                    '--agent.run-name',name,'--agent.logger','tensorboard','--agent.resume','False',
                    '--agent.upload-model','False','--checkpoint',str(checkpoint),'--gpu-ids',f'[{args.gpu}]']
                run_logged(command,HERE/f'manual_t{c}_{tag}.log')
                runs=list((ROOT/'logs/rsl_rl/k1_mimic').glob('*_'+name));assert len(runs)==1,runs
                run=runs[0]
            else:run=checkpoint.parent
            assert (run/'model_9999.pt').is_file()
            out=HERE/'exports'/f't{c}_targeted_v3_9999_manual_{tag}';out.parent.mkdir(exist_ok=True)
            run_logged([sys.executable,str(HERE/'export_snapshot.py'),str(run),str(out),'model_9999.pt'],
                HERE/f'manual_export_t{c}_{tag}.log',env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            state['completed'].append(dict(chapter=c,run=str(run),export=str(out),quality_status='awaiting_sim2sim_evaluation'))
            status.write_text(json.dumps(state,indent=2))
        state['status']='complete'
    except BaseException as exc:
        state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed';state['error']=repr(exc)
        raise
    finally:status.write_text(json.dumps(state,indent=2))
if __name__=='__main__':main()
