# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Warm start from SoftEntry v2; resume only matching restored-DR checkpoints."""
from copy import deepcopy
import torch
from mjlab.rl import MjlabOnPolicyRunner
from .curriculum_runner import StandUpCurriculumRunner
from .soft_entry_runner import SoftEntryRunner


def randomization_spec(cfg):
  events = cfg.events
  specs = {}
  fields = {
    'physics_material': ('low_range', 'regular_range', 'low_probability_start',
      'low_probability_end', 'ramp_steps', 'curriculum_command'),
    'joint_home_offset_noise': ('pos_distribution_params', 'operation'),
    'torso_com_offset_noise': ('com_range',),
  }
  for name, keys in fields.items():
    if name in events:
      term = events[name]
      specs[name] = {'mode': term.mode, **{k: deepcopy(term.params[k]) for k in keys}}
  return {'profile': 'original_stand_up_dr_v1', 'events': specs,
    'resample_material_on_reset': cfg.commands['reference_trajectory'].resample_material_on_reset}


class SoftEntryDRRunner(SoftEntryRunner):
  def entry_spec(self):
    return {**super().entry_spec(), 'version': 3,
      'restored_randomization': randomization_spec(self.env.unwrapped.cfg)}

  def save(self, path, infos=None):
    StandUpCurriculumRunner.save(self, path, {**(infos or {}),
      'ai_sapiens_entry_version': 3, 'ai_sapiens_entry_spec': self.entry_spec()})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    metadata = torch.load(path, map_location='cpu', weights_only=False).get('infos') or {}
    version = metadata.get('ai_sapiens_entry_version')
    saved = metadata.get('ai_sapiens_entry_spec', {})
    current = self.entry_spec()
    if version not in (2, 3) or saved.get('motion_sha256') != current['motion_sha256']:
      raise ValueError('SoftEntry-DR requires a matching SoftEntry v2 or DR v3 checkpoint')
    if load_cfg is not None:
      # Explicit actor-only playback must not restore curriculum or optimizer.
      infos = MjlabOnPolicyRunner.load(self, path, load_cfg, strict, map_location)
    elif version == 3:
      if saved != current:
        raise ValueError('SoftEntry-DR checkpoint contract/randomization differs from this task')
      infos = StandUpCurriculumRunner.load(self, path, None, strict, map_location)
      self.current_learning_iteration += 1
      self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
    else:
      # v2's trained weights are useful, but its optimizer, iteration and
      # no-DR curriculum do not represent the new training run.
      infos = MjlabOnPolicyRunner.load(self, path, {'actor': True, 'critic': True}, strict, map_location)
      self.current_learning_iteration = 0
      self.env.unwrapped.common_step_counter = 0
      c = self.env.unwrapped.command_manager.get_term('reference_trajectory')
      c.load_curriculum_state_dict(dict(difficulty_stage=0, attempts=0, successes=0,
        last_success_rate=0., completed_successes=0, failed_attempts=0, interrupted_attempts=0))
      self.env.reset()
      print('[INFO] SoftEntry-DR warm start: v2 actor/critic loaded; new optimizer, iteration 0, curriculum stage 0.')
    self._refresh()
    return infos
