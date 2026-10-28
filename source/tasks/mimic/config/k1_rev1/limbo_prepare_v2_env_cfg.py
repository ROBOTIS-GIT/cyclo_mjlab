from dataclasses import fields
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.mdp import limbo_prepare_v1 as v1,limbo_prepare_v2 as v2
from .limbo_prepare_v1_env_cfg import k1_rev1_limbo_prepare_v1_env_cfg,k1_rev1_limbo_prepare_v1_runner_cfg

def k1_rev1_limbo_prepare_v2_env_cfg(play=False):
  cfg=k1_rev1_limbo_prepare_v1_env_cfg(play=play)
  old=cfg.commands['reference_trajectory']
  values={f.name:getattr(old,f.name) for f in fields(old)}
  values.update(initial_alpha_max=1.,curriculum_steps=1,ready_foot_error=.07)
  cfg.commands['reference_trajectory']=v2.LimboPreparationV2CommandCfg(**values)
  for group in ('actor','critic'):
    cfg.observations[group].terms['limbo_preparation_state']=ObservationTermCfg(func=v2.preparation_state)
  # A stationary failed preparation must not keep collecting tracking rewards.
  for name,term in list(cfg.rewards.items()):
    if term.weight>0 and name not in ('limbo_full_success',) and not name.startswith('prepare_'):
      cfg.rewards[name]=RewardTermCfg(func=v2.motion_only,weight=term.weight,params={'original_func':term.func,'original_params':term.params})
  cfg.rewards.pop('prepare_posture');cfg.rewards.pop('prepare_stillness')
  cfg.rewards['prepare_progress']=RewardTermCfg(func=v2.preparation_progress,weight=1.)
  cfg.rewards['prepare_time']=RewardTermCfg(func=v1.preparation_time_cost,weight=-1.)
  # Reward manager scales by 20ms: +5 once on release, -10 on preparation failure.
  cfg.rewards['prepare_reached']=RewardTermCfg(func=v1.preparation_reached,weight=250.)
  cfg.rewards['prepare_failed']=RewardTermCfg(func=v2.preparation_failed,weight=-500.)
  cfg.rewards['prepare_episode_record']=RewardTermCfg(func=v2.record_episode,weight=1.)
  return cfg

def k1_rev1_limbo_prepare_v2_runner_cfg():
  cfg=k1_rev1_limbo_prepare_v1_runner_cfg()
  cfg.experiment_name='k1_limbo_1p2_prepare_v2'
  cfg.run_name='bar120_prepare_v2_from6200'
  cfg.algorithm.learning_rate=3e-5
  cfg.algorithm.schedule='fixed'
  cfg.max_iterations=5000
  return cfg
