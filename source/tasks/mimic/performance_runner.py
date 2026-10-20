# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Separate Koryo performance warm-start and resume semantics."""
import torch
from mjlab.rl import MjlabOnPolicyRunner
from .motion_curriculum import motion_identity, validate_checkpoint_motion
from .koryo_schedule import migrate_state


class KoryoPerformanceRunner(MjlabOnPolicyRunner):
  motion_name = "koryo"
  checkpoint_version = 1
  disturbance_curriculum = False

  def _restore_difficulty(self, infos):
    if self.disturbance_curriculum:
      cmd = self.env.unwrapped.command_manager.get_term('reference_trajectory')
      cmd.load_curriculum_state_dict(migrate_state(infos.get(self.motion_name + '_curriculum')))
    elif infos.get(self.motion_name + '_performance', {}).get('disturbance_curriculum'):
      raise ValueError('Resume this checkpoint with the ContactV2-Curriculum task.')
    elif infos.get(self.motion_name + '_performance', {}).get('fixed_disturbance_stage') is not None:
      raise ValueError('Resume this checkpoint with the ContactV2-Max task.')

  def resumes_performance_state(self, state):
    if not state:
      return False
    version = state.get('version')
    if version not in (1, self.checkpoint_version):
      raise ValueError('Unsupported performance checkpoint version.')
    return version == self.checkpoint_version

  def load(self,path,load_cfg=None,strict=True,map_location=None):
    if load_cfg is not None:
      return super().load(path,load_cfg,strict,map_location)
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    infos = checkpoint.get('infos') or {}
    validate_checkpoint_motion(infos, self.motion_name)
    resume = self.resumes_performance_state(infos.get(self.motion_name + '_performance'))
    if resume and self.checkpoint_version == 2:
      steps = infos.get('contact_training_steps')
      if type(steps) is not int or steps < 0:
        raise ValueError('Invalid ContactV2 training step counter.')
    # A changed reward needs a fresh optimizer and a small fixed learning rate.
    # Retain both trained networks, but never inherit mastered/iteration state.
    loaded = super().load(path,None if resume else {'actor':True,'critic':True},strict,map_location)
    if resume:
      self.current_learning_iteration += 1
      self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
      if self.checkpoint_version == 2:
        self.env.unwrapped.common_step_counter = int(infos['contact_training_steps'])
    else:
      self.current_learning_iteration = 0
      self.env.unwrapped.common_step_counter = 0
      with torch.no_grad():
        # Keep the learned mean policy; lower fine-tuning exploration only.
        for name,parameter in self.alg.actor.named_parameters():
          if name=='distribution.log_std_param':
            parameter.clamp_(max=-2.302585093)  # std <= 0.1
      print('[INFO] Koryo performance warm-start: networks restored, optimizer and iteration reset.')
    self._restore_difficulty(infos)
    self.env.reset()
    cmd = self.env.unwrapped.command_manager.get_term('reference_trajectory')
    scores = infos.get('performance_start_sampler') if resume else None
    if scores is not None:
      if scores.shape!=cmd.failure_score_by_bin.shape or not bool(scores.isfinite().all()) or bool((scores<0).any()):
        raise ValueError('Invalid performance sampler state.')
      cmd.failure_score_by_bin.copy_(scores.to(cmd.device))
    cmd.frame_ids -= 1
    cmd._update_command()
    return loaded

  def learn(self,num_learning_iterations,init_at_random_ep_len=False):
    return super().learn(num_learning_iterations,init_at_random_ep_len=False)

  def save(self,path,infos=None):
    infos = dict(infos or {})
    cmd = self.env.unwrapped.command_manager.get_term('reference_trajectory')
    infos['motion_identity'] = motion_identity(self.motion_name,cmd.cfg.trajectory_file)
    infos[self.motion_name + '_performance'] = {'version':self.checkpoint_version,'objective':'foot_contact_quality','fixed_noise_scale':0.25,'push_scale':0.0}
    if self.disturbance_curriculum:
      infos[self.motion_name + '_curriculum'] = cmd.curriculum_state_dict()
      infos[self.motion_name + '_performance'] = {
        'version': self.checkpoint_version, 'objective': 'foot_contact_quality',
        'disturbance_curriculum': True,
      }
    if self.checkpoint_version == 2:
      infos['contact_training_steps'] = self.env.unwrapped.common_step_counter
    if getattr(self, 'fixed_disturbance_stage', None) is not None:
      infos[self.motion_name + '_curriculum'] = cmd.curriculum_state_dict()
      infos[self.motion_name + '_performance'] = {
        'version': self.checkpoint_version, 'objective': 'foot_contact_quality',
        'fixed_disturbance_stage': self.fixed_disturbance_stage,
        'fixed_noise_scale': 1.0, 'push_scale': 1.0,
      }
    infos['performance_start_sampler'] = cmd.failure_score_by_bin.detach().cpu().clone()
    super().save(path,infos)


class KoryoContactV2Runner(KoryoPerformanceRunner):
  """V1 -> V2 starts a new reward phase; V2 -> V2 resumes it exactly."""
  checkpoint_version = 2


class KoryoContactCurriculumRunner(KoryoContactV2Runner):
  """Keep ContactV2 PPO/ramp/sampler state while restoring the evaluated stage."""
  disturbance_curriculum = True

  def resumes_performance_state(self, state):
    if not state or state.get('version') != 2:
      raise ValueError('Contact curriculum requires a ContactV2 checkpoint.')
    return True


class KoryoContactMaxRunner(KoryoContactV2Runner):
  """Scratch training at full disturbances; resume only this experiment type."""
  fixed_disturbance_stage = 8

  def resumes_performance_state(self, state):
    if (not state or state.get('version') != 2
        or state.get('fixed_disturbance_stage') != self.fixed_disturbance_stage):
      raise ValueError('Full-disturbance task requires a matching checkpoint; omit --checkpoint for scratch training.')
    return True

  def _restore_difficulty(self, infos):
    state = migrate_state(infos.get(self.motion_name + '_curriculum'))
    if state['difficulty_stage'] != self.fixed_disturbance_stage:
      raise ValueError('Full-disturbance checkpoint must retain stage 8.')
    cmd = self.env.unwrapped.command_manager.get_term('reference_trajectory')
    cmd.load_curriculum_state_dict(state)


class GeumgangContactMaxRunner(KoryoContactMaxRunner):
  """Reuse ContactV2-Max while keeping Geumgang checkpoint identity separate."""
  motion_name = "geumgang"
