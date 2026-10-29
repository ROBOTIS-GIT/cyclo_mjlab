"""Like Jennie reference with the selected Taegeuk 4 reward configuration."""
from pathlib import Path

from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.mdp.performance_contacts import add_foot_sensors
from source.tasks.mimic.mdp.kpop_support import SupportStability
from source.tasks.mimic.mdp.kpop_arms import ArmRecovery
from source.tasks.mimic.mdp.kpop_taeguek4_rewards import (
    KickPreservation, SidekickSupport, arm_pose_tracking, extra_grf, soft_contact,
)
from .taeguek_env_cfg import k1_rev1_taeguek_env_cfg
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg

MOTION = (Path(__file__).resolve().parents[4] / 'assets/motions/K1_rev1/'
          'like_jennie_early_plus10/like_jennie_50hz.npz')


def k1_rev1_like_jennie_env_cfg(play=False):
    # Chapter 4 supplies the selected run's base terms, including pelvis XY.
    cfg = add_foot_sensors(k1_rev1_taeguek_env_cfg(4, play))
    command = cfg.commands['reference_trajectory']
    command.trajectory_file = str(MOTION)
    command.first_frame_env_fraction = .2 if not play else 0.
    command.start_sampling_uniform_mix = .2
    cfg.episode_length_s = 52.4 if not play else int(1e9)
    cfg.scene.num_envs = 4096 if not play else 1
    cfg.rewards['t6_support'] = RewardTermCfg(func=SupportStability,weight=-1.,params={'ramp_steps':1})
    cfg.rewards['t6_arm_pose'] = RewardTermCfg(func=arm_pose_tracking,weight=.4,params={})
    cfg.rewards['t6_arm_recovery'] = RewardTermCfg(func=ArmRecovery,weight=-1.,params={
        'joint_weight':.35,'elbow_weight':.15,'hand_weight':.30,'orientation_weight':.10,
        'hold_velocity_weight':.08,'hold_acceleration_weight':.01,'ramp_steps':1})
    cfg.rewards['t6_kick_height'] = RewardTermCfg(func=KickPreservation,weight=-.25,params={'ramp_steps':1})
    cfg.rewards['t6_soft_contact'] = RewardTermCfg(func=soft_contact,weight=-.15,params={'ramp_steps':1})
    cfg.rewards['t4_grf_extra'] = RewardTermCfg(func=extra_grf,weight=-.15,params={'ramp_steps':1})
    # Exact selected Taegeuk-4 reward, including its original time mask.
    cfg.rewards['t4_sidekick_stability'] = RewardTermCfg(func=SidekickSupport,weight=-1.,params={'ramp_steps':72000})
    return cfg


def k1_rev1_like_jennie_ppo_cfg():
    # The source settings were a low-rate fine-tune of a Taegeuk checkpoint.
    # Use the standard from-scratch PPO settings when learning a new dance.
    cfg = k1_rev1_mimic_ppo_runner_cfg()
    cfg.run_name = 'like_jennie_taeguek04_rewards_scratch'
    cfg.resume = False
    return cfg
