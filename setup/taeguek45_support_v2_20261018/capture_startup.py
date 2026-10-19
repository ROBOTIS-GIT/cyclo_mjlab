from pathlib import Path
from datetime import datetime,timezone
import re,json,subprocess
root=Path('/home/ubuntu/cyclo_mjlab');setup=root/'setup/taeguek45_support_v2_20261018'
log=(setup/'driver.log').read_text()
assert not (setup/'driver.exit').exists(),log[-5000:]
assert 'Traceback (most recent call last)' not in log
iterations=re.findall(r'Learning iteration (\d+)/10000',log)
assert iterations and int(iterations[-1])>=10,iterations
runs=list((root/'logs/rsl_rl/k1_mimic').glob('*taeguek04_support_v2_finetune_20261018_support_v2'))
assert len(runs)==1 and (runs[0]/'model_0.pt').is_file()
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),chapter=4,
  current_iteration=int(iterations[-1]),target_additional_updates=10000,
  sequence=[4,5],run=str(runs[0]),checkpoint0=True,
  driver_running=True,gpu_status=subprocess.check_output(['nvidia-smi',
    '--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).splitlines())
(setup/'startup_verified.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
