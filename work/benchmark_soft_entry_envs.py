from pathlib import Path
import subprocess, os, signal, time, json, re, statistics, datetime
root=Path('/home/robotis-insu/cyclo_mjlab')
folder=root/'work/soft_entry_env_benchmark';folder.mkdir(exist_ok=True)
pid=3895756
cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
assert 'train.py Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry' in cmdline
assert '--env.scene.num-envs 512' in cmdline
checkpoint=root/'logs/rsl_rl/k1_mimic/2026-10-19_15-03-14_umr_stand_up_ai_sapiens_soft_entry/model_0.pt'
python='/home/robotis-insu/miniconda3/envs/k1_rl_mjlab/bin/python'
results=[]; child=None
start=datetime.datetime.now().astimezone().isoformat()
try:
  os.kill(pid,signal.SIGSTOP)
  print('Production training paused with SIGSTOP; optimizer/environment remain in memory.',flush=True)
  time.sleep(2)
  for count in (512,2048):
    log=folder/f'benchmark_{count}.log'
    command=[python,'-u','scripts/reinforcement_learning/train.py',
      'Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry',
      '--checkpoint',str(checkpoint),'--env.scene.num-envs',str(count),
      '--agent.max-iterations','40','--agent.run-name',f'benchmark_soft_entry_{count}']
    peak_mem=0;util=[];temperatures=[]
    t=time.monotonic()
    with log.open('wb') as f:
      child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      print(f'Benchmark {count} started: PID {child.pid}',flush=True)
      while child.poll() is None:
        if time.monotonic()-t>240: raise TimeoutError(f'{count} benchmark exceeded 240 seconds')
        out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        for row in out.splitlines():
          cols=row.split(',')
          if int(cols[0])==child.pid:peak_mem=max(peak_mem,int(cols[1]))
        out=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True)
        v=[int(s) for s in out.strip().split(',')];util.append(v[0]);temperatures.append(v[1])
        time.sleep(1)
    text=log.read_text()
    if child.returncode: raise RuntimeError(f'{count} failed with code {child.returncode}: {text[-2000:]}')
    durations=[float(s) for s in re.findall(r'Iteration time:\s+([0-9.]+)s',text)]
    assert len(durations)==40, len(durations)
    stable=durations[10:]
    record=dict(num_envs=count,iterations=len(durations),warmup_discarded=10,
      median_iteration_seconds=statistics.median(stable),mean_iteration_seconds=statistics.mean(stable),
      samples_per_iteration=count*24,samples_per_second=count*24/statistics.mean(stable),
      peak_process_gpu_memory_mib=peak_mem,max_gpu_temperature_c=max(temperatures),
      run_directory=re.search(r'Logging experiment in directory: (.+)',text)[1],log=str(log),
      elapsed_including_startup_seconds=time.monotonic()-t)
    results.append(record)
    (folder/'results_partial.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    child=None
finally:
  if child is not None and child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try: child.wait(timeout=10)
    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
  os.kill(pid,signal.SIGCONT)
  record=dict(started_at=start,finished_at=datetime.datetime.now().astimezone().isoformat(),
    production_pid=pid,production_resumed=True,checkpoint=str(checkpoint),benchmarks=results)
  (folder/'results.json').write_text(json.dumps(record,indent=2)+'\n')
  print('Production training resumed with SIGCONT.',flush=True)
