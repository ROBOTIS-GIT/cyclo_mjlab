"""Small GPU integration check for both refinement tasks and reset/cache behavior."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import source.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
results=[]
for chapter in (4,5):
  cfg=load_env_cfg(f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-SupportV2')
  cfg.scene.num_envs=32;cfg.seed=42
  env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
  tc=env.reward_manager.get_term_cfg('taeguek_support_quality');term=tc.func
  mask=term.reference_support;fps=float(term.cmd.reference.fps)
  windows=[(14.63,14.70,0),(4.735,4.92,1)] if chapter==4 else [(4.81,4.87,1)]
  coverage=[]
  for a,b,f in windows:
    fraction=mask[round(a*fps):round(b*fps),f].float().mean().item()
    assert fraction==1.,(chapter,a,b,f,fraction)
    coverage.append(dict(start=a,end=b,foot=f,eligible=fraction))
  assert term.reference_kick.any()
  assert not (term.reference_kick & mask).any()
  env.reset()
  for step in range(64):
    obs,reward,terminated,truncated,info=env.step(torch.zeros(32,env.action_manager.total_action_dim,device='cuda:0'))
    assert torch.isfinite(reward).all()
    cost=term(env,**tc.params).clone();age=term.tracker.expected_age.clone()
    again=term(env,**tc.params)
    torch.testing.assert_close(cost,again,rtol=0,atol=0)
    torch.testing.assert_close(age,term.tracker.expected_age,rtol=0,atol=0)
    assert torch.isfinite(cost).all() and (cost>=0).all()
  # Reset grace remains per-environment and prevents artificial spawn penalties.
  others=term.age[2:].clone();term.reset(torch.tensor([0,1],device=env.device))
  assert (term.age[:2]==0).all()
  torch.testing.assert_close(others,term.age[2:],rtol=0,atol=0)
  assert (term.tracker.expected_age[:2]==0).all()
  result=dict(chapter=chapter,finite_steps=64,coverage=coverage,
    support_frames=mask.sum(0).tolist(),kick_frames=term.reference_kick.sum(0).tolist(),
    cache_and_partial_reset=True)
  results.append(result);print('V2_CHECK_RESULT '+json.dumps(result),flush=True)
  env.close();del env,term;torch.cuda.empty_cache()
Path('setup/taeguek45_support_v2_20261018/preflight.json').write_text(json.dumps(results,indent=2))
