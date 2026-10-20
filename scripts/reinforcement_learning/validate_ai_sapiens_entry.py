"""Headless deployment-contract and full-clip integration check on CUDA."""
from pathlib import Path
import sys, json, argparse
from dataclasses import asdict
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import source.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--checkpoint',required=True)
  parser.add_argument('--steps',type=int,default=600)
  args=parser.parse_args()
  task='Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens'
  cfg=load_env_cfg(task,play=True); cfg.scene.num_envs=8
  env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
  wrapped=RslRlVecEnvWrapper(env,clip_actions=10.)
  runner=load_runner_cls(task)(wrapped,asdict(load_rl_cfg(task)),device='cuda:0')
  runner.load(args.checkpoint,load_cfg={'actor':True},map_location='cuda:0')
  policy=runner.get_inference_policy(device='cuda:0')
  c=env.command_manager.get_term('reference_trajectory')
  a=env.action_manager.get_term('joint_pos')
  assert c.frame_ids.eq(0).all()
  assert env.action_manager.action.eq(0).all()
  obs=wrapped.get_observations()
  assert obs['actor'].shape==(8,124) and obs['critic'].shape==(8,256)
  zeros=0; min_gap=1.; max_delta=0.; all_rewards=[]
  with torch.inference_mode():
    for step in range(args.steps):
      before=c.frame_ids.clone()
      actor=policy(obs)
      obs,reward,done,_=wrapped.step(actor)
      assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
      reset=c.frame_ids.eq(0)
      zeros+=int(reset.sum())
      if reset.any():
        assert env.action_manager.action[reset].eq(0).all()
        assert c.elapsed[reset].eq(0).all()
      if step < 15:
        assert c.frame_ids.eq(step+1).all(),(step,c.frame_ids)
        s=(step*.02+.018)/.3
        expected=s*s*(3-2*s)
        torch.testing.assert_close(c.metrics['handoff_alpha'],torch.full((8,),expected,device='cuda:0'),atol=2e-5,rtol=0.)
        torch.testing.assert_close(env.action_manager.action,actor.clamp(-10,10))
      if step==15:
        torch.testing.assert_close(c.metrics['handoff_alpha'],torch.ones(8,device='cuda:0'))
      if step<75:
        min_gap=min(min_gap,float(c.metrics['entry_foot_clearance_m'].min()))
        max_delta=max(max_delta,float(a.target_delta.abs().max()))
      all_rewards.append(float(reward.mean()))
  print('VALIDATION_RESULT '+json.dumps(dict(steps=args.steps,envs=8,actor=124,critic=256,
    resets=zeros,successes=c.completed_successes,failures=c.failed_attempts,
    min_initial_proxy_clearance_m=min_gap,max_initial_target_delta_rad=max_delta,
    mean_reward=sum(all_rewards)/len(all_rewards),finite=True)))
  env.close()

if __name__=='__main__': main()
