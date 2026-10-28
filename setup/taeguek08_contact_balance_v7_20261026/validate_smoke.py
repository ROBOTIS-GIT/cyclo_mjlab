from pathlib import Path
import json,sys,copy
import torch,yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from config import MARKER,revision,completed_updates,SOURCE_SHA,TOTAL
runs=sorted((ROOT/'logs/rsl_rl/k1_mimic').glob('*_'+MARKER+'_smoke'))
first=[p for p in runs if (p/'model_7.pt').is_file()][-1]
second=[p for p in runs if (p/'model_15.pt').is_file()][-1]
a=torch.load(first/'model_7.pt',map_location='cpu',weights_only=False);b=torch.load(second/'model_15.pt',map_location='cpu',weights_only=False)
assert completed_updates(a,4096,True)==8 and completed_updates(b,4096,True)==16
assert a['infos']['live_training_health']['live_environments_over24']>0
assert b['infos']['live_training_health']['live_environments_over24']>0
assert a['infos']['env_state']['common_step_counter']==192
assert b['infos']['env_state']['common_step_counter']==384
assert b['infos'][MARKER]['source_sha256']==SOURCE_SHA
assert b['infos'][MARKER]['initial_state']=='warmstart'
for k in ('actor_state_dict','critic_state_dict'):
 assert all(torch.isfinite(x).all() for x in b[k].values())
 assert any(not torch.equal(v,b[k][n]) for n,v in a[k].items())
steps=lambda d:sorted({float(v['step']) for v in d['optimizer_state_dict']['state'].values() if 'step' in v})
assert min(steps(b))>max(steps(a))
for num,smoke in [(64,True),(4096,False)]:
 try:completed_updates(b,num,smoke)
 except ValueError:pass
 else:raise AssertionError('Resume guard did not reject mismatch')
bad=copy.deepcopy(b);bad['infos']['env_state']['common_step_counter']+=1
try:completed_updates(bad,4096,True)
except ValueError:pass
else:raise AssertionError('Counter mismatch accepted')
class TupleSafeLoader(yaml.SafeLoader):
    pass
TupleSafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node: tuple(loader.construct_sequence(node)))
agent=yaml.load((second/'params/agent.yaml').read_text(),Loader=TupleSafeLoader)
assert agent['algorithm']['learning_rate']==3e-6 and agent['algorithm']['schedule']=='fixed'
assert agent['algorithm']['entropy_coef']==.00015
result=dict(status='passed',revision=revision(),main_started=False,main_budget=TOTAL,source_sha256=SOURCE_SHA,
 initial_smoke=str(first),resumed_smoke=str(second),first_completed=8,resumed_completed=16,first_counter=192,resumed_counter=384,
 optimizer_steps_first=steps(a),optimizer_steps_resumed=steps(b),resume_mismatch_rejections=['num_envs','smoke/main','step counter'],
 interpretation='Plumbing/finite-update/resume verification only, not motion-quality validation',
 num_envs=4096,current_container_gpu='Existing cyclo_mjlab container GPU verified healthy; Desktop RTX 5090')
(HERE/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
