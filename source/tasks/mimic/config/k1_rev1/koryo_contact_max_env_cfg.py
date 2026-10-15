# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""ContactV2 with full disturbances from the first reset and original PPO settings."""
from dataclasses import dataclass, fields

from source.tasks.mimic.koryo_schedule import initial_state
from source.tasks.mimic.mdp.koryo_curriculum import (
  KoryoCurriculumCommandCfg, KoryoObservationNoise,
)
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from .koryo_contact_v2_env_cfg import k1_rev1_koryo_contact_v2_env_cfg


@dataclass(kw_only=True)
class KoryoFullDisturbanceCommandCfg(KoryoCurriculumCommandCfg):
  def build(self, env):
    command = super().build(env)
    command.load_curriculum_state_dict(initial_state(8))
    return command


def k1_rev1_koryo_contact_max_env_cfg(play=False):
  cfg = k1_rev1_koryo_contact_v2_env_cfg(play=play)
  # Isolate the added contact terms: retain the original action-rate penalty.
  cfg.rewards['action_rate_l2'].weight = -0.1
  if not play:
    old = cfg.commands['reference_trajectory']
    cfg.commands['reference_trajectory'] = KoryoFullDisturbanceCommandCfg(**{
      f.name: getattr(old, f.name) for f in fields(old) if f.init
    })
    cfg.curriculum['contact_max_difficulty'] = cfg.curriculum.pop('performance_fixed_difficulty')
    for term in cfg.observations['actor'].terms.values():
      if isinstance(term.noise, KoryoObservationNoise):
        term.noise.curriculum_scale = 1.0
  return cfg


def k1_rev1_koryo_contact_max_ppo_cfg():
  cfg = k1_rev1_mimic_ppo_runner_cfg()
  cfg.max_iterations = 60000
  cfg.run_name = 'koryo_contact_max_from_scratch'
  return cfg
