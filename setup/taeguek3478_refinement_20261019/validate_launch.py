"""Validate immutable inputs, meaningful PPO updates, and configuration before launch."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
import torch
from refinement import env_cfg,agent_cfg
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in json.loads((HERE/'code_manifest.json').read_text()).items():
    assert sha(HERE/name)==digest,name
assert [r['chapter'] for r in json.loads((HERE/'preflight.json').read_text())]==[3,4,7,8]
health=json.loads((HERE/'rollout_health.json').read_text())
assert [r['chapter'] for r in health]==[3,4,7,8]
assert all(r['longest_live_episode_steps']>64 and r['max_targeted_cost']>0 for r in health)
results=[]
for c,s in json.loads((HERE/'sources.json').read_text()).items():
    c=int(c);cfg=env_cfg(c);agent=agent_cfg()
    assert agent.algorithm.learning_rate==1e-4 and agent.algorithm.schedule=='fixed'
    path=Path(s['source_run'])/s['checkpoint'];assert sha(path)==s['checkpoint_sha256']
    data=torch.load(path,map_location='cpu',weights_only=False)
    smoke=list((ROOT/'logs/rsl_rl/k1_mimic').glob(f'*taeguek{c:02d}_targeted_v3_smoke_20261019/model_1.pt'))
    assert len(smoke)==1,smoke
    output=torch.load(smoke[0],map_location='cpu',weights_only=False)
    marker=output['infos']['targeted_3478']
    assert marker['sha256']==sha(path) and marker['fresh_optimizer']
    assert output['infos']['env_state']['common_step_counter']==48
    assert all(torch.isfinite(v).all() for v in output['actor_state_dict'].values())
    assert any(not torch.equal(v,output['actor_state_dict'][k]) for k,v in data['actor_state_dict'].items())
    rates=[g['lr'] for g in output['optimizer_state_dict']['param_groups']]
    assert all(x==1e-4 for x in rates),rates
    motion=Path(cfg.commands['reference_trajectory'].trajectory_file)
    assert sha(motion.with_suffix('.csv'))==s['csv_sha256'], 'CSV differs from installed sim2sim motion'
    results.append(dict(chapter=c,source=str(path),sha256=sha(path),source_iteration=int(data['iter']),
        motion=str(motion),motion_sha256=sha(motion),csv_sha256=sha(motion.with_suffix('.csv')),
        smoke=str(smoke[0]),fresh_optimizer=True,learning_rate=1e-4,new_updates=10000))
(HERE/'launch_validation.json').write_text(json.dumps(results,indent=2))
print('LAUNCH_VALIDATED',json.dumps(results),flush=True)
