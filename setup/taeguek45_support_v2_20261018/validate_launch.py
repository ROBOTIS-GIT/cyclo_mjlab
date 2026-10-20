"""Fail closed before starting either expensive job."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import source.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
root=Path('/workspace/cyclo_mjlab');audit=root/'setup/taeguek45_support_v2_20261018'
for rel,sha in json.loads((audit/'source_manifest.json').read_text()).items():
  assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha,rel
assert len(json.loads((audit/'preflight.json').read_text()))==2
results=[]
for chapter,iteration in ((4,29999),(5,15000)):
  task=f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-SupportV2'
  cfg=load_env_cfg(task);agent=load_rl_cfg(task)
  assert agent.algorithm.learning_rate==1e-4 and agent.algorithm.schedule=='fixed'
  runs=list((root/'logs/rsl_rl/k1_mimic').glob(f'*taeguek{chapter:02d}_support_v1_scratch_20261016_support_v1'))
  assert len(runs)==1
  checkpoint=runs[0]/f'model_{iteration}.pt';data=torch.load(checkpoint,map_location='cpu',weights_only=False)
  assert data['iter']==iteration
  assert all(torch.isfinite(v).all() for v in data['actor_state_dict'].values())
  smoke=list((root/'logs/rsl_rl/k1_mimic').glob(f'*taeguek{chapter:02d}_support_v2_smoke_20261018/model_1.pt'))
  assert len(smoke)==1,smoke
  smoke_data=torch.load(smoke[0],map_location='cpu',weights_only=False)
  metadata=smoke_data['infos']['taeguek_support_v2']
  digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
  assert metadata['source_iteration']==iteration and metadata['sha256']==digest
  assert metadata['fresh_optimizer'] and metadata['learning_rate']==1e-4
  assert smoke_data['infos']['env_state']['common_step_counter']==48
  assert all(torch.isfinite(v).all() for v in smoke_data['actor_state_dict'].values())
  assert any(not torch.equal(v,smoke_data['actor_state_dict'][k]) for k,v in data['actor_state_dict'].items())
  motion=Path(cfg.commands['reference_trajectory'].trajectory_file)
  results.append(dict(chapter=chapter,source=str(checkpoint),sha256=digest,
    source_iteration=iteration,motion=str(motion),motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),
    smoke=str(smoke[0]),new_updates=10000,learning_rate=1e-4))
(audit/'launch_validation.json').write_text(json.dumps(results,indent=2))
print('LAUNCH_VALIDATED '+json.dumps(results),flush=True)
