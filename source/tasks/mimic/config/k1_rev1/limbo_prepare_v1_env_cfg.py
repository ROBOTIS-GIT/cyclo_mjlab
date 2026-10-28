"""120cm preparation training, isolated from all existing limbo tasks."""
from dataclasses import fields
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from source import SRC_PATH
from source.tasks.mimic.mdp import limbo_prepare_v1 as prep
from .limbo_1p2_v2_env_cfg import k1_rev1_limbo_1p2_v2_env_cfg,k1_rev1_limbo_1p2_v2_runner_cfg

def k1_rev1_limbo_prepare_v1_env_cfg(play=False):
  cfg=k1_rev1_limbo_1p2_v2_env_cfg(play=play)
  cfg.episode_length_s=11.0
  old=cfg.commands['reference_trajectory']
  ref=prep.LimboPreparationCommandCfg(**{f.name:getattr(old,f.name) for f in fields(old)})
  ref.reset_bank_file=str(SRC_PATH/'assets/motions/K1_rev1/limbo_1p2_prepare_v1/reset_bank.npz')
  cfg.commands['reference_trajectory']=ref
  for group in ('actor','critic'):
    cfg.observations[group].terms['limbo_phase_contacts']=ObservationTermCfg(func=prep.phase_and_contacts)
  cfg.rewards['limbo_full_success']=RewardTermCfg(func=prep.completion_reward,weight=10.)
  cfg.rewards['prepare_posture']=RewardTermCfg(func=prep.preparation_posture,weight=2.)
  cfg.rewards['prepare_stillness']=RewardTermCfg(func=prep.preparation_stillness,weight=1.)
  cfg.rewards['prepare_time']=RewardTermCfg(func=prep.preparation_time_cost,weight=-.25)
  cfg.rewards['prepare_reached']=RewardTermCfg(func=prep.preparation_reached,weight=10.)
  cfg.rewards['limbo_stance_slip'].func=prep.stance_motion
  cfg.rewards['limbo_stance_rotation'].func=prep.stance_motion
  cfg.terminations['prepare_timeout']=TerminationTermCfg(func=prep.preparation_timeout)
  return cfg

def k1_rev1_limbo_prepare_v1_runner_cfg():
  cfg=k1_rev1_limbo_1p2_v2_runner_cfg()
  cfg.experiment_name='k1_limbo_1p2_prepare_v1'
  cfg.run_name='bar120_prepare_v1_from120'
  cfg.algorithm.learning_rate=1e-4
  cfg.save_interval=100
  cfg.max_iterations=10000
  return cfg
