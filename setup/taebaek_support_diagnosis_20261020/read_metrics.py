from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json,statistics
root=Path('logs/rsl_rl/k1_mimic'); out={}
for name in ['2026-10-20_07-52-13_taebaek_support_v1_finetune_20261020_075204','2026-10-20_02-38-55_taebaek_arm_refined_manual']:
 ea=EventAccumulator(str(root/name),size_guidance={'scalars':0});ea.Reload();rows={}
 for tag in ea.Tags()['scalars']:
  vals=ea.Scalars(tag)
  windows=[(0,100),(400,500),(900,1000),(1900,2000),(2900,3000),(3900,4000),(4900,5000),(vals[-1].step-99,vals[-1].step+1)] if 'support_v1' in name else [(16900,17000)]
  bins=[]
  for lo,hi in windows:
   v=[x.value for x in vals if lo<=x.step<hi]
   if v:bins.append({'lo':lo,'hi':hi,'mean':statistics.mean(v)})
  rows[tag]={'last_step':vals[-1].step,'bins':bins}
 out[name]=rows
Path('setup/taebaek_support_diagnosis_20261020/metrics.json').write_text(json.dumps(out,indent=2))
for name,rows in out.items():
 print('\nRUN',name)
 for k,v in rows.items():
  print(k, v['last_step'], ' '.join(f"{b['lo']}:{b['mean']:.5g}" for b in v['bins']))
