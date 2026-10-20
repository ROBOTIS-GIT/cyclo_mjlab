from pathlib import Path
import json, hashlib, shutil, torch, re, datetime
root=Path('/home/robotis-insu/cyclo_mjlab')
record=json.loads((root/'work/soft_entry_training/launch.json').read_text())
log=Path(record['log']).read_text()
run=root/re.search(r'Logging experiment in directory: (.+)',log)[1]
x=torch.load(run/'model_0.pt',map_location='cpu',weights_only=False)
assert x['infos']['ai_sapiens_entry_version']==2
assert x['iter']==0 and x['optimizer_state_dict']['param_groups'][0]['lr']==5e-5
for n,d in x.items():
  if isinstance(d,dict):
    for k,v in d.items():
      if torch.is_tensor(v): assert torch.isfinite(v).all(),(n,k)
assert (run/'params/k1_0008_stand_up_ai_sapiens_soft_entry.csv').is_file()
files=['source/tasks/mimic/mdp/soft_entry.py','source/tasks/mimic/mdp/ai_sapiens_entry.py',
 'source/tasks/mimic/soft_entry_runner.py',
 'source/tasks/mimic/config/k1_rev1/umr_stand_up_soft_entry_env_cfg.py',
 'source/tasks/mimic/config/k1_rev1/umr_stand_up_ai_sapiens_env_cfg.py',
 'scripts/reinforcement_learning/build_ai_sapiens_soft_entry_motion.py',
 'scripts/reinforcement_learning/train_soft_entry.sh','tests/test_soft_entry.py',
 'scripts/reinforcement_learning/validate_soft_entry.py', 'docs/stand_up_soft_entry.md']
snapshot=run/'params/training_sources';snapshot.mkdir(exist_ok=True)
hashes={}
for f in files:
  dest=snapshot/f;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(root/f,dest)
  hashes[f]=hashlib.sha256(dest.read_bytes()).hexdigest()
record.update(run_dir=str(run),verified_at=datetime.datetime.now().astimezone().isoformat(),
  first_checkpoint=str(run/'model_0.pt'),first_checkpoint_finite=True,
  latest_iteration=int(re.findall(r'Learning iteration (\d+)/',log)[-1]),
  eta=re.findall(r'ETA: (\S+)',log)[-1],source_hashes=hashes)
(run/'params/launch_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
(root/'work/soft_entry_training/verified_launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'},indent=2))
