"""Distinguish completed-episode logging from actual per-step termination counts."""
from pathlib import Path
from dataclasses import asdict
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
import torch
import source.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from refinement import env_cfg,agent_cfg,TargetedRunner
configure_torch_backends();results=[]
for c,s in json.loads((HERE/'sources.json').read_text()).items():
 cfg=env_cfg(int(c));cfg.scene.num_envs=32;cfg.seed=42
 raw=ManagerBasedRlEnv(cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw)
 runner=TargetedRunner(env,asdict(agent_cfg()),None,device='cuda:0');runner.load(str(Path(s['source_run'])/s['checkpoint']))
 actor=runner.get_inference_policy(device='cuda:0');obs=env.get_observations()
 term=raw.reward_manager.get_term_cfg('targeted_turn' if c=='3' else 'taeguek_support_quality').func
 deaths=[];lengths=[];costs=[]
 for i in range(128):
  with torch.inference_mode():action=actor(obs)
  obs,reward,done,info=env.step(action)
  assert torch.isfinite(reward).all()
  deaths.append(int(done.sum()));lengths.append(int(raw.episode_length_buf.max()))
  costs.append(float((term.cached if c=='3' else term.geom_cached).max()))
 assert max(lengths)>64,(c,lengths)
 assert sum(deaths[1:])<32*10,(c,sum(deaths))
 assert max(costs)>0,(c,costs)
 result=dict(chapter=int(c),steps=128,initial_reset_terminations=deaths[0],later_terminations=sum(deaths[1:]),
  longest_live_episode_steps=max(lengths),max_targeted_cost=max(costs),every_step_terminated=False)
 results.append(result);print('HEALTH_RESULT',json.dumps(result),flush=True)
 env.close();del raw,env,runner;torch.cuda.empty_cache()
(HERE/'rollout_health.json').write_text(json.dumps(results,indent=2))
