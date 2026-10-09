# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Koryo-only curriculum; the original Koryo task remains a baseline."""

from dataclasses import fields

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.utils.noise import UniformNoiseCfg
from source.tasks.mimic.mdp.koryo_curriculum import (
  KoryoCurriculumCommandCfg,
  KoryoObservationNoise,
  koryo_com_offset,
  koryo_home_offset,
  koryo_material,
  koryo_push,
  update_koryo_difficulty,
)
from .koryo_env_cfg import k1_rev1_koryo_env_cfg


def k1_rev1_koryo_curriculum_env_cfg(play=False):
  cfg = k1_rev1_koryo_env_cfg(play=play)
  if play:
    return cfg
  old = cfg.commands["reference_trajectory"]
  command = KoryoCurriculumCommandCfg(**{
    f.name: getattr(old, f.name) for f in fields(old) if f.init
  })
  command.first_frame_env_fraction = 0.1
  command.start_sampling_uniform_mix = 0.25
  cfg.commands["reference_trajectory"] = command
  cfg.curriculum["koryo_difficulty"] = CurriculumTermCfg(func=update_koryo_difficulty)
  for name, func in (
    ("physics_material", koryo_material),
    ("torso_com_offset_noise", koryo_com_offset),
    ("joint_home_offset_noise", koryo_home_offset),
  ):
    cfg.events[name].func = func
    cfg.events[name].mode = "reset"
  cfg.events["push_robot"].func = koryo_push
  for term in cfg.observations["actor"].terms.values():
    if isinstance(term.noise, UniformNoiseCfg):
      term.noise = KoryoObservationNoise(
        n_min=term.noise.n_min, n_max=term.noise.n_max,
        operation=term.noise.operation,
      )
  return cfg
