"""Bounded full-ramp rollout; no learning, no production checkpoint."""
from pathlib import Path
from dataclasses import asdict
import sys,os,json,tempfile
os.environ.setdefault('MPLCONFIGDIR','/tmp/t7-v4-validation-mpl')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from config import env_cfg,agent_cfg,ContactRunner,SOURCE,revision
from landing import WEIGHTS
configure_torch_backends();torch.set_num_threads(2)
cfg=env_cfg();cfg.scene.num_envs=64
cfg.commands['reference_trajectory'].focus_windows=((26.9,27.1),);cfg.commands['reference_trajectory'].focus_env_fraction=.5
cfg.rewards['targeted_right_landing'].params['ramp_steps']=1
assert cfg.scale_rewards_by_dt
env=ManagerBasedRlEnv(cfg,device='cuda:0');env.reset();wrap=RslRlVecEnvWrapper(env)
a=agent_cfg();a.run_name='taeguek07_landing_v4_full_validation_smoke'
r=ContactRunner(wrap,asdict(a),tempfile.mkdtemp(prefix='t7-v4-full-'),'cuda:0');r.load(SOURCE)
term=env.reward_manager.get_term_cfg('targeted_right_landing').func
pi=r.get_inference_policy('cuda:0');obs=wrap.get_observations();peaks={};max_error=0.
for _ in range(128):
 with torch.inference_mode():obs,rewards,dones,extras=wrap.step(pi(obs))
 assert torch.isfinite(rewards).all() and torch.isfinite(term.cached).all()
 metrics=term.cmd.metrics
 for key,val in metrics.items():
  if key.startswith('t7_landing_'):
   assert torch.isfinite(val).all(),key
   peaks[key]=max(peaks.get(key,0.),float(val.max()))
 # Metrics clone survives resets; the full ramp should equal the weighted new
 # terms on every active step. Exclude the initial ramp=0 observation.
 if float(metrics['t7_landing_ramp'].min())==1:
  total=sum(WEIGHTS[k]*metrics['t7_landing_'+k] for k in WEIGHTS)
  error=float((total-metrics['t7_landing_cost']).abs().max());max_error=max(max_error,error)
  torch.testing.assert_close(total,metrics['t7_landing_cost'],atol=1e-5,rtol=1e-5)
assert peaks['t7_landing_ramp']==1
for k in ['braking','total_peak','total_rate','total_excess']:assert peaks['t7_landing_'+k]>0,k
with torch.inference_mode():
 x=term(env).clone();torch.testing.assert_close(x,term(env),rtol=0,atol=0)
 term.reset(torch.tensor([0],device='cuda:0'));assert term.age[0]==0 and term.last_frame[0]==-1
result=dict(status='passed',revision=revision(),device='cuda:0',num_envs=64,steps=128,full_ramp=True,weighted_cost_max_error=max_error,metrics_max=peaks,source_warmstart_exact=True,reset_cache_checked=True,main_started=False)
(HERE/'validation_full.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));env.close()
