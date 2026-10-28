"""Targeted full-strength runtime validation, not a policy-quality evaluation."""
from pathlib import Path
from dataclasses import asdict
import json,sys,os,tempfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR','/tmp/t8-v7-mpl')
import torch,yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from source.utils import export_sim2real_cfg
from config import env_cfg,agent_cfg,ContactRunner,SOURCE,revision
from refinement_costs import WEIGHTS
from precision_costs import WEIGHTS as PRECISION_WEIGHTS
from balance_costs import WEIGHTS as BALANCE_WEIGHTS

configure_torch_backends();torch.set_num_threads(2)
cfg=env_cfg();cfg.scene.num_envs=64
cfg.rewards['t8_contact_precision'].params['ramp_steps']=1
cfg.rewards['t8_contact_balance'].params['ramp_steps']=1
assert cfg.scale_rewards_by_dt
env=ManagerBasedRlEnv(cfg,device='cuda:0')
cmd=env.command_manager.get_term('reference_trajectory')
starts=torch.tensor([330,460,900,1875,1935,2075],device='cuda:0')
def sample(ids):
    ids=torch.as_tensor(ids,device='cuda:0',dtype=torch.long)
    cmd.frame_ids[ids]=starts[ids%6]
cmd._sample_start_frames=sample
env.reset();wrap=RslRlVecEnvWrapper(env)
agent=agent_cfg();agent.run_name='taeguek08_contact_balance_v7_full_validation_smoke'
log=tempfile.mkdtemp(prefix='t8-v7-full-')
runner=ContactRunner(wrap,asdict(agent),log,'cuda:0');runner.load(SOURCE)
export_sim2real_cfg(env,Path(log))
actual=yaml.safe_load((Path(log)/'params/sim2real.yaml').read_text())
source=yaml.safe_load((HERE/'source/params/sim2real.yaml').read_text())
assert actual==source,'Deployment parameters changed from selected source'
precision=env.reward_manager.get_term_cfg('t8_contact_precision').func
assert precision.locked.any(), 'No stationary reference support found'
balance=env.reward_manager.get_term_cfg('t8_contact_balance').func
coverage={}
for foot,lo,hi in [(1,7.88,8.54),(1,41.82,42.92),(0,32.4,33.4)]:
    ix=(balance.time>=lo)&(balance.time<hi)
    value=float((balance.gates[ix,foot]>0).float().mean())
    coverage[str(lo)]=value
    assert value>.75,(lo,value)
term=env.reward_manager.get_term_cfg('t8_landing_support_refinement').func
policy=runner.get_inference_policy('cuda:0');obs=wrap.get_observations();peaks={};active_steps=0
for _ in range(180):
    with torch.inference_mode():obs,rewards,dones,extras=wrap.step(policy(obs))
    assert torch.isfinite(rewards).all()
    metrics=cmd.metrics
    for k,v in metrics.items():
        if k.startswith(('t8_v5_','t8_v6_','t8_v7_')):
            assert torch.isfinite(v).all(),k
            peaks[k]=max(peaks.get(k,0.),float(v.max()))
    if float(metrics['t8_v5_ramp'].min())==1:
        total=sum(WEIGHTS[k]*metrics['t8_v5_'+k] for k in WEIGHTS)
        torch.testing.assert_close(total,metrics['t8_v5_cost'],atol=1e-5,rtol=1e-5)
    if float(metrics['t8_v6_ramp'].min())==1:
        total=sum(PRECISION_WEIGHTS[k]*metrics['t8_v6_'+k] for k in PRECISION_WEIGHTS)
        torch.testing.assert_close(total,metrics['t8_v6_cost'],atol=1e-5,rtol=1e-5)
    if float(metrics['t8_v7_ramp'].min())==1:
        total=sum(BALANCE_WEIGHTS[k]*metrics['t8_v7_'+k] for k in BALANCE_WEIGHTS)
        torch.testing.assert_close(total,metrics['t8_v7_cost'],atol=1e-5,rtol=1e-5)
    active_steps+=int((metrics['t8_v5_active']>0).any())
assert active_steps>20
for k in ['total_peak','total_rate','total_excess','foot_braking','pelvis_braking','quiet_drift','quiet_speed']:
    assert peaks['t8_v5_'+k]>0,k
for k in PRECISION_WEIGHTS:
    assert peaks['t8_v6_'+k]>0,k
for k in BALANCE_WEIGHTS:
    assert peaks['t8_v7_'+k]>0,k
with torch.inference_mode():
    a=balance(env).clone();torch.testing.assert_close(a,balance(env),rtol=0,atol=0)
    balance.reset(torch.tensor([0],device='cuda:0'))
    assert balance.age[0]==0 and balance.last_frame[0]==-1
    a=precision(env).clone();torch.testing.assert_close(a,precision(env),rtol=0,atol=0)
    precision.reset(torch.tensor([0],device='cuda:0'))
    assert precision.age[0]==0 and precision.last_frame[0]==-1
    assert torch.all(precision.offset[0]==0) and not precision.previous[0].any()
    a=term(env).clone();torch.testing.assert_close(a,term(env),rtol=0,atol=0)
    term.reset(torch.tensor([0],device='cuda:0'))
    assert term.age[0]==0 and term.last_frame[0]==-1
    assert torch.all(term.phase_offset[0]==0) and not term.previous[0].any()
result=dict(status='passed',revision=revision(),device='cuda:0',environments=64,steps=180,
            target_start_times_s=[6.6,9.2,18.,37.5,38.7,41.5],new_support_coverage=coverage,locked_reference_frames_per_foot=precision.locked.sum(0).tolist(),active_steps=active_steps,metrics_max=peaks,
            identical_source_deployment_config=True,weighted_cost_checked=True,cache_reset_checked=True,
            main_started=False,interpretation='Runtime validation only; no learned motion quality claim')
(HERE/'validation_full.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2));env.close()
