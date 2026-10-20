from pathlib import Path
import json, subprocess, datetime, hashlib
import torch
root=Path('/home/robotis-insu/cyclo_mjlab')
smoke=root/'logs/rsl_rl/k1_mimic/2026-10-19_15-02-26_soft_entry_smoke/model_2.pt'
x=torch.load(smoke,map_location='cpu',weights_only=False)
assert x['iter']==2 and x['infos']['ai_sapiens_entry_version']==2
assert x['optimizer_state_dict']['param_groups'][0]['lr']==5e-5
for name, value in x.items():
  if isinstance(value,dict):
    for key,t in value.items():
      if torch.is_tensor(t): assert torch.isfinite(t).all(),(name,key)
now=datetime.datetime.now().astimezone()
folder=root/'work/soft_entry_training'; folder.mkdir(exist_ok=True)
log=folder/('training_'+now.strftime('%Y%m%d_%H%M%S')+'.log')
cmd=['bash','scripts/reinforcement_learning/train_soft_entry.sh']
with log.open('wb') as f:
  p=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,started_at=now.isoformat(),cwd=str(root),command=cmd,log=str(log),
  task='Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry',num_envs=512,iterations=10000,learning_rate=5e-5,
  checkpoint=str(root/'logs/rsl_rl/k1_mimic/2026-10-19_10-15-20_umr_stand_up_ai_sapiens/model_9999.pt'),
  tests='2 new + 9 existing unit tests; 8/8 deterministic full-clip integration; 3 PPO iterations and finite checkpoint tensors')
(folder/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
