from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataclasses import asdict
import torch
import source.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from refinement import env_cfg,agent_cfg,TargetedRunner,k1_rev1_taeguek_support_env_cfg
configure_torch_backends();HERE=Path(__file__).resolve().parent
sources=json.loads((HERE/'sources.json').read_text())
for mode in ['base','targeted']:
 cfg=k1_rev1_taeguek_support_env_cfg(3) if mode=='base' else env_cfg(3)
 cfg.scene.num_envs=32;cfg.seed=42
 raw=ManagerBasedRlEnv(cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw)
 runner=TargetedRunner(env,asdict(agent_cfg()),None,device='cuda:0')
 runner.load(str(Path(sources['3']['source_run'])/sources['3']['checkpoint']))
 actor=runner.get_inference_policy(device='cuda:0');obs=env.get_observations()
 cmd=raw.command_manager.get_term('reference_trajectory')
 for i in range(3):
  with torch.inference_mode():action=actor(obs)
  print('DIAG',mode,'step',i,'frame',cmd.frame_ids.tolist(),'body_error0',(cmd.aligned_body_position_w[0,:,2]-cmd.robot_body_pos_w[0,:,2]).tolist(),'action_absmax',action.abs().max().item(),flush=True)
  obs,reward,done,extra=env.step(action)
  print('DIAG_AFTER',mode,'done',done.tolist(),'terms',{k:v.sum().item() for k,v in raw.termination_manager._term_dones.items()},flush=True)
 env.close();del env,raw,runner;torch.cuda.empty_cache()
