"""Small GPU integration check for all four support tasks; no training mutation."""
import sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import source.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
results=[]
for chapter in range(2,6):
 cfg=load_env_cfg(f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-SupportV1')
 cfg.scene.num_envs=32;cfg.seed=42
 env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
 term=env.reward_manager.get_term_cfg('taeguek_support_quality').func
 mask=term.reference_support;old=term.original_reference_support
 fps=float(term.cmd.reference.fps)
 assert mask.any() and (mask<=old).all()
 report={'chapter':chapter,'frames':len(mask),'fps':fps,'old_support_frames':old.sum(0).tolist(),
         'new_support_frames':mask.sum(0).tolist(),'excluded_frames':(old&~mask).sum(0).tolist()}
 if chapter==2:
  def frac(a,b,f):return mask[round(a*fps):round(b*fps),f].float().mean().item()
  report['kick_left_23_66_24_24']=frac(23.66,24.24,0)
  report['relocation_left_9_74_9_90']=frac(9.74,9.90,0)
  report['relocation_right_7_22_7_52']=frac(7.22,7.52,1)
  assert report['kick_left_23_66_24_24']>.9,report
  assert report['relocation_left_9_74_9_90']<.1,report
  assert report['relocation_right_7_22_7_52']<.1,report
 torch.save({'support':mask.cpu(),'old_support':old.cpu(),'bottom':term.reference_bottom.cpu(),
             'lowpoint_speed':term.reference_lowpoint_speed.cpu()},
            f'setup/taeguek_support_v1_20261016/labels_{chapter:02d}.pt')
 env.reset()
 for step in range(64):
  obs,reward,terminated,truncated,info=env.step(torch.zeros(32,env.action_manager.total_action_dim,device='cuda:0'))
  assert torch.isfinite(reward).all()
  cost=term(env)
  assert torch.isfinite(cost).all() and (cost>=0).all()
 report['finite_steps']=64
 results.append(report);print('CHECK_RESULT '+json.dumps(report),flush=True)
 env.close();del term,env;torch.cuda.empty_cache()
Path('setup/taeguek_support_v1_20261016/preflight.json').write_text(json.dumps(results,indent=2))
