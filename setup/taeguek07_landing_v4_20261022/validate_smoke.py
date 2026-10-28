from pathlib import Path
import json,sys,copy
import torch,yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from config import MARKER,revision,completed_updates,SOURCE_SHA,TOTAL
CPU='--cpu' in sys.argv
N=64 if CPU else 4096
SUFFIX='_cpu_smoke' if CPU else '_smoke'
runs=sorted((ROOT/'logs/rsl_rl/k1_mimic').glob('*_'+MARKER+SUFFIX))
first=[p for p in runs if (p/'model_7.pt').is_file()][-1]
second=[p for p in runs if (p/'model_15.pt').is_file()][-1]
a=torch.load(first/'model_7.pt',map_location='cpu',weights_only=False);b=torch.load(second/'model_15.pt',map_location='cpu',weights_only=False)
assert completed_updates(a,N,True)==8 and completed_updates(b,N,True)==16
assert a['infos']['env_state']['common_step_counter']==192
assert b['infos']['env_state']['common_step_counter']==384
assert b['infos'][MARKER]['source_sha256']==SOURCE_SHA
assert b['infos'][MARKER]['initial_state']=='warmstart'
for k in ('actor_state_dict','critic_state_dict'):
 assert all(torch.isfinite(x).all() for x in b[k].values())
 assert any(not torch.equal(v,b[k][n]) for n,v in a[k].items())
steps=lambda d:sorted({float(v['step']) for v in d['optimizer_state_dict']['state'].values() if 'step' in v})
assert min(steps(b))>max(steps(a))
for num,smoke in [(2048,True),(N,False)]:
 try:completed_updates(b,num,smoke)
 except ValueError:pass
 else:raise AssertionError('Resume guard did not reject mismatch')
bad=copy.deepcopy(b);bad['infos']['env_state']['common_step_counter']+=1
try:completed_updates(bad,N,True)
except ValueError:pass
else:raise AssertionError('Counter mismatch accepted')
class TupleSafeLoader(yaml.SafeLoader):
    pass
TupleSafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node: tuple(loader.construct_sequence(node)))
agent=yaml.load((second/'params/agent.yaml').read_text(),Loader=TupleSafeLoader)
assert agent['algorithm']['learning_rate']==5e-6 and agent['algorithm']['schedule']=='fixed'
assert agent['algorithm']['entropy_coef']==.0003
import math,re
assert b['infos']['guard_window']
assert all(all(math.isfinite(float(v)) for v in row.values()) for row in b['infos']['guard_window'])
source_cfg=yaml.safe_load((HERE/'source/params/sim2real.yaml').read_text())
current_cfg=yaml.safe_load((second/'params/sim2real.yaml').read_text())
assert source_cfg==current_cfg, 'Control/observation/action interface changed'
text=(HERE/('cpu_smoke.log' if CPU else 'smoke.log')).read_text()
for name in ['active','cost','braking','peak','early_height','total_peak','total_rate','total_excess']:
    samples=[float(v) for v in re.findall(r'Metrics/reference_trajectory/t7_landing_'+name+r': ([0-9.eE+-]+)',text)]
    assert samples and all(math.isfinite(v) for v in samples) and max(samples)>0,name
result=dict(status='passed',revision=revision(),main_started=False,main_budget=TOTAL,source_sha256=SOURCE_SHA,
 initial_smoke=str(first),resumed_smoke=str(second),first_completed=8,resumed_completed=16,first_counter=192,resumed_counter=384,
 optimizer_steps_first=steps(a),optimizer_steps_resumed=steps(b),tests=32,targeted_reward_exercised=True,control_interface_unchanged=True,guard_telemetry_valid=True,resume_mismatch_rejections=['num_envs','smoke/main','step counter'],
 interpretation='Plumbing/finite-update/resume verification only, not motion-quality validation',
 validation_device='cpu' if CPU else 'cuda', num_envs=N)
(HERE/('validation_cpu.json' if CPU else 'validation.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
