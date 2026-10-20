from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json,statistics
root=Path('logs/rsl_rl/k1_mimic');out={}
for name in ['2026-10-20_02-38-55_taebaek_arm_refined_manual','2026-10-19_13-58-40_taebaek_contact_corrected_scratch50000_after_t7_20261019']:
 ea=EventAccumulator(str(root/name),size_guidance={'scalars':0});ea.Reload();tags=ea.Tags()['scalars'];rows={}
 for tag in tags:
  if not any(k in tag.lower() for k in ['reward','length','error_','termination','learning_rate','iteration_time']):continue
  vs=ea.Scalars(tag);data=[]
  for lo,hi in ([(9000,10000),(13000,14000),(16000,17000),(17000,18000)] if 'manual' in name else [(49000,50000)]):
   v=[x.value for x in vs if lo<=x.step<hi]
   if v:data.append({'start':lo,'end':hi,'samples':len(v),'mean':statistics.mean(v)})
  rows[tag]={'last_step':vs[-1].step,'last_value':vs[-1].value,'last_wall_time':vs[-1].wall_time,'bins':data}
 out[name]={'tags':tags,'metrics':rows}
Path('setup/taebaek_progress_audit_20261020/trends.json').write_text(json.dumps(out,indent=2));print('Saved scalar trends')
