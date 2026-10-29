"""Bounded reward refinement of the Like Jennie baseline task."""
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.mdp.like_jennie_tracking_rewards import joint_tracking_score, world_foot_tracking_score
from .like_jennie_env_cfg import k1_rev1_like_jennie_env_cfg, k1_rev1_like_jennie_ppo_cfg


def like_jennie_tracking_env_cfg(play=False):
    cfg=k1_rev1_like_jennie_env_cfg(play)
    cfg.rewards['reference_pelvis_xy'].weight=1.5
    cfg.rewards['reference_pelvis_xy'].params['std']=0.35
    cfg.rewards['like_jennie_joint_tracking']=RewardTermCfg(func=joint_tracking_score,weight=0.6,params={'std':0.35})
    cfg.rewards['like_jennie_world_foot_tracking']=RewardTermCfg(func=world_foot_tracking_score,weight=0.5,params={'std':0.25})
    return cfg


def like_jennie_tracking_ppo_cfg():
    cfg=k1_rev1_like_jennie_ppo_cfg()
    cfg.run_name='like_jennie_tracking_refine_v1'
    cfg.algorithm.learning_rate=1e-4
    cfg.algorithm.schedule='fixed'
    cfg.algorithm.entropy_coef=0.001
    cfg.max_iterations=500
    cfg.save_interval=100
    return cfg


def like_jennie_tracking_v2_env_cfg(play=False):
    cfg=like_jennie_tracking_env_cfg(play)
    cfg.rewards['reference_pelvis_xy'].weight=1.25
    cfg.rewards['reference_pelvis_xy'].params['std']=0.5
    cfg.rewards['like_jennie_joint_tracking'].weight=0.4
    cfg.rewards['like_jennie_world_foot_tracking'].weight=0.2
    return cfg


def like_jennie_tracking_v2_ppo_cfg():
    cfg=like_jennie_tracking_ppo_cfg()
    cfg.run_name='like_jennie_tracking_refine_v2'
    cfg.algorithm.learning_rate=3e-5
    cfg.max_iterations=200
    cfg.save_interval=100
    return cfg
