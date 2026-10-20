"""Register a process-local TaeBaek XY task and use the standard trainer."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import source.tasks
from mjlab.tasks.registry import register_mjlab_task
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.config.k1_rev1.base_env_cfg import k1_rev1_mimic_env_cfg
from source.tasks.mimic.config.k1_rev1.agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from source.tasks.mimic.mdp.pelvis_xy_rewards import reference_pelvis_xy

TASK='Cyclo-Mimic-K1-Rev1-Taebaek-XY'
# 100-FPS time axis; contact-corrected UMR with local arm jitter/clearance refinement.
# Preserve accelerated timing (47.18 s); no global smoothing or speed caps.
MOTION=ROOT/'source/assets/motions/K1_rev1/taebaek/k1_taebaek_arm_refined_50hz_converted.npz'

def env_cfg(play=False):
  cfg=k1_rev1_mimic_env_cfg(play=play)
  cfg.commands['reference_trajectory'].trajectory_file=str(MOTION)
  cfg.rewards['reference_pelvis_xy']=RewardTermCfg(
    func=reference_pelvis_xy,weight=1.,params={'command_name':'reference_trajectory','std':.5})
  return cfg

agent=k1_rev1_mimic_ppo_runner_cfg()
agent.run_name='taebaek_arm_refined_20261020'
agent.resume=False
agent.upload_model=False
register_mjlab_task(task_id=TASK,env_cfg=env_cfg(),play_env_cfg=env_cfg(play=True),rl_cfg=agent)

if __name__=='__main__':
  if sys.argv[1:]==['--check']:
    import torch,numpy as np
    assert MOTION.is_file(),MOTION
    with np.load(MOTION,allow_pickle=False) as motion:
      assert float(motion['fps'][0])==50
      assert motion['joint_pos'].shape[1]==23
      assert all(np.isfinite(motion[k]).all() for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'])
      frames=len(motion['joint_pos'])
    assert torch.cuda.is_available(),'No CUDA device visible in this container'
    assert torch.ones(1,device='cuda:0').item()==1
    cfg=env_cfg();assert abs(cfg.sim.mujoco.timestep*cfg.decimation-.02)<1e-9
    print(json.dumps({'task':TASK,'gpu':torch.cuda.get_device_name(0),
      'motion':str(MOTION),'frames':frames,'fps':50,'num_envs':cfg.scene.num_envs,
      'max_iterations':agent.max_iterations,'resume':agent.resume,
      'reward':'standard Mimic + pelvis XY; no SupportV1/V2 additions','training_started':False},indent=2))
  else:
    from scripts.reinforcement_learning.train import main
    sys.argv.insert(1,TASK)
    main()
