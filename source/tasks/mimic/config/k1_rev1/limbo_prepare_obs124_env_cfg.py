from dataclasses import fields
import copy
from mjlab.managers.reward_manager import RewardTermCfg
from source import SRC_PATH
from source.tasks.mimic.mdp import limbo_prepare_v2 as v2,limbo_prepare_v3 as v3,limbo_prepare_obs124 as v4
from .limbo_prepare_v2_env_cfg import k1_rev1_limbo_prepare_v2_env_cfg,k1_rev1_limbo_prepare_v2_runner_cfg

OBSERVATION_ORDER=('motion_command','motion_anchor_ori_b','base_ang_vel','joint_pos_rel','joint_vel_rel','last_action')

def k1_rev1_limbo_prepare_obs124_env_cfg(play=False):
    cfg=k1_rev1_limbo_prepare_v2_env_cfg(play=play)
    cfg.scene.num_envs=1 if play else 4096
    actor_terms=cfg.observations['actor'].terms
    assert all(name in actor_terms for name in OBSERVATION_ORDER)
    for group in ('actor','critic'):
        cfg.observations[group].terms={name:copy.deepcopy(actor_terms[name]) for name in OBSERVATION_ORDER}
    old=cfg.commands['reference_trajectory']
    values={f.name:getattr(old,f.name) for f in fields(old)}
    values['trajectory_file']=str(SRC_PATH/'assets/motions/K1_rev1/limbo_1p2_obs124/limbo.npz')
    cfg.commands['reference_trajectory']=v4.Obs124PreparationCommandCfg(**values)
    cfg.episode_length_s=11.
    for name,term in list(cfg.rewards.items()):
        if term.func is v2.motion_only:
            cfg.rewards[name]=RewardTermCfg(func=term.params['original_func'],weight=term.weight,params=term.params['original_params'])
    for name in ('prepare_time','prepare_progress','prepare_failed','prepare_reached'):
        cfg.rewards.pop(name,None)
    cfg.rewards['prepare_orientation']=RewardTermCfg(func=v3.preparation_orientation_cost,weight=-2.)
    cfg.rewards['prepare_stability']=RewardTermCfg(func=v3.preparation_stability_cost,weight=-.2)
    cfg.rewards['prepare_reached']=RewardTermCfg(func=v4.prefix_reached,weight=250.)
    cfg.rewards['prepare_failed']=RewardTermCfg(func=v4.prefix_failed,weight=-250.)
    cfg.rewards['prepare_episode_record']=RewardTermCfg(func=v4.record_episode,weight=1.)
    cfg.rewards['limbo_full_success']=RewardTermCfg(func=v4.completion_reward,weight=50.)
    cfg.terminations.pop('prepare_timeout',None)
    return cfg

def k1_rev1_limbo_prepare_obs124_runner_cfg():
    cfg=k1_rev1_limbo_prepare_v2_runner_cfg()
    cfg.experiment_name='k1_limbo_1p2_prepare_obs124'
    cfg.run_name='bar120_obs124_fixed2s'
    cfg.actor.class_name='MLPModel'
    cfg.actor.distribution_cfg={'class_name':'GaussianDistribution','init_std':.02,'std_type':'log'}
    cfg.algorithm.learning_rate=5e-6
    cfg.algorithm.schedule='fixed'
    cfg.algorithm.entropy_coef=0.
    cfg.algorithm.clip_param=.1
    cfg.max_iterations=20000
    cfg.save_interval=100
    return cfg
