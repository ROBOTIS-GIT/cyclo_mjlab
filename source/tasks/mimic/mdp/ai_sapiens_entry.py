# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Train the unchanged ai_sapiens 0.3 s entry controller, with a moving reference."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.utils.lab_api.math import quat_apply
from source import SRC_PATH
from .commands import ReferenceTrajectoryCommand
from .stand_up_curriculum import StandUpCurriculumCommand, StandUpCurriculumCommandCfg, curriculum_push
from .ready_handoff import ReadyHandoffAction, ReadyHandoffActionCfg

ASSET = SRC_PATH / 'assets/robots/ai_sapiens_entry'
CONTRACT = json.loads((ASSET / 'contract.json').read_text())

@dataclass(kw_only=True)
class AiSapiensCommandCfg(StandUpCurriculumCommandCfg):
  deterministic_bank: bool = False
  entry_penalty_seconds: float = 1.5

  def build(self, env):
    if not 0 < self.first_frame_env_fraction <= 1 or self.entry_penalty_seconds <= 0:
      raise ValueError('Invalid entry sampling/penalty configuration')
    return AiSapiensCommand(self, env)

class AiSapiensCommand(StandUpCurriculumCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    with np.load(ASSET / 'ready_bank.npz', allow_pickle=False) as bank:
      for key, file in (('contract_sha256', 'contract.json'), ('model_sha256', 'mujoco/k1/k1.xml')):
        if str(bank[key]) != hashlib.sha256((ASSET / file).read_bytes()).hexdigest():
          raise ValueError('Ready bank is stale; regenerate it after changing the deployment snapshot')
      names = bank['joint_names'].tolist()
      order = [names.index(name) for name in self.robot.joint_names]
      self.bank = {k: torch.as_tensor(bank[k], device=self.device) for k in
        ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel')}
      for key in ('joint_pos', 'joint_vel'):
        self.bank[key] = self.bank[key][:, order]
      if not all(torch.isfinite(v).all() for v in self.bank.values()):
        raise ValueError('Nonfinite ready bank')
    self.enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.penalty_mask = self.enabled.clone()
    self.just_reset = self.enabled.clone()
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    for key in ('handoff_alpha', 'handoff_target_delta_rad', 'handoff_torque_delta_nm',
                'handoff_peak_target_delta_rad', 'handoff_peak_torque_delta_nm', 'entry_foot_clearance_m'):
      self.metrics[key] = torch.zeros_like(self.elapsed)
    # ai_sapiens' CSV loader uses a forward difference (and repeats the last).
    self.reference_joint_velocity = torch.diff(self.reference.joint_position, dim=0) / env.step_dt
    self.reference_joint_velocity = torch.cat((self.reference_joint_velocity, self.reference_joint_velocity[-1:]), 0)

  @property
  def joint_vel(self):
    return self.reference_joint_velocity[self.frame_ids]

  def _resample_command(self, env_ids):
    ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    if not len(ids):
      return
    super()._resample_command(ids)
    self.enabled[ids] = (ids < int(self.num_envs * self.cfg.first_frame_env_fraction)) | self.cfg.start_from_zero
    selected = ids[self.enabled[ids]]
    sample = torch.zeros(len(selected), device=self.device, dtype=torch.long) if self.cfg.deterministic_bank else torch.randint(len(self.bank['root_pose']), (len(selected),), device=self.device)
    if len(selected):
      pose = self.bank['root_pose'][sample].clone()
      pose[:, :3] += self._env.scene.env_origins[selected]
      self.robot.write_joint_state_to_sim(self.bank['joint_pos'][sample], self.bank['joint_vel'][sample], env_ids=selected)
      self.robot.write_root_link_pose_to_sim(pose, env_ids=selected)
      self.robot.write_root_link_velocity_to_sim(self.bank['root_velocity'][sample], env_ids=selected)
      self.robot.clear_state(env_ids=selected)
    # Clip-end resets happen after env.step's forward pass: refresh kinematics
    # so the next observation describes the new pose, not the previous episode.
    self._env.sim.forward()
    self.elapsed[ids] = 0
    self.just_reset[ids] = True
    self.penalty_mask[ids] = self.enabled[ids]
    for key, value in self.metrics.items():
      if key.startswith('handoff_'):
        value[ids] = 0
    self._env.action_manager.reset(ids)
    self._env.action_manager.get_term('joint_pos').reset_handoff(ids)

  def _update_command(self):
    # env.step resets terminated worlds immediately before command.compute.
    # Those worlds must expose frame zero to their first actor inference.
    self.frame_ids -= self.just_reset.long()
    self.just_reset.zero_()
    super()._update_command()
    # A clip-end reset inside super already exposes frame zero this step.
    self.just_reset.zero_()

  def refresh_reference(self):
    self.frame_ids -= 1
    ReferenceTrajectoryCommand._update_command(self)
    self.just_reset.zero_()

@dataclass(kw_only=True)
class AiSapiensActionCfg(ReadyHandoffActionCfg):
  def build(self, env):
    return AiSapiensAction(self, env)

class AiSapiensAction(ReadyHandoffAction):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    indices = [CONTRACT['joints'].index(j) for j in self.target_names]
    ready = torch.tensor(CONTRACT['ready_target'], device=self.device)[indices]
    if not torch.allclose(self.ready_target, ready, atol=1e-5):
      raise ValueError('Reference frame zero no longer matches deployment ReadyPose')
    self.ready_target.copy_(ready)
    self.actor_target = self.applied_target.clone()

  def process_actions(self, actions):
    # Keep raw policy output for last_action; blend only the actuator target.
    JointPositionAction.process_actions(self, actions)
    self.actor_target.copy_(self._processed_actions)
    self.previous_target.copy_(self.applied_target)
    c = self.command
    c.just_reset.zero_()  # a real actor step now owns the new episode clock.
    c.penalty_mask = c.enabled & (c.elapsed < c.cfg.entry_penalty_seconds)

  def apply_actions(self):
    c = self.command
    s = (c.elapsed / CONTRACT['duration']).clamp(0., 1.)
    alpha = torch.where(c.enabled, s.square() * (3. - 2. * s), 1.)[:, None]
    self.applied_target.copy_(torch.lerp(self.ready_target.expand_as(self.actor_target), self.actor_target, alpha))
    self.target_delta.copy_(self.applied_target - self.previous_target)
    self._processed_actions.copy_(self.applied_target)
    kp = torch.lerp(self.ready_kp.expand_as(self.actor_target), self.policy_kp.expand_as(self.actor_target), alpha)
    kd = torch.lerp(self.ready_kd.expand_as(self.actor_target), self.policy_kd.expand_as(self.actor_target), alpha)
    model = self._env.sim.model
    model.actuator_gainprm[:, self.ctrl_ids, 0] = kp
    model.actuator_biasprm[:, self.ctrl_ids, 1] = -kp
    model.actuator_biasprm[:, self.ctrl_ids, 2] = -kd
    c.metrics['handoff_alpha'][:] = alpha[:, 0]
    JointPositionAction.apply_actions(self)
    c.elapsed += self._env.physics_dt


def entry_push(env, env_ids, velocity_range):
  c = env.command_manager.get_term('reference_trajectory')
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  ids = ids[~c.penalty_mask[ids]]
  if len(ids):
    curriculum_push(env, ids, velocity_range)


def entry_foot_clearance(env, radius=.055, margin=.01):
  """Conservative sole-volume proxy; actual foot collision spheres are too small."""
  c = env.command_manager.get_term('reference_trajectory')
  robot = c.robot
  ids = [robot.body_names.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]
  points = torch.tensor([[-.05, 0, -.035], [0, 0, -.035], [.05, 0, -.035], [.10, 0, -.035]], device=env.device)
  quat = robot.data.body_link_quat_w[:, ids, None, :].expand(-1, -1, len(points), -1)
  xyz = robot.data.body_link_pos_w[:, ids, None, :] + quat_apply(quat, points.expand(env.num_envs, 2, -1, -1))
  gap = torch.linalg.vector_norm(xyz[:, 0, :, None] - xyz[:, 1, None, :], dim=-1).flatten(1).amin(1) - 2 * radius
  c.metrics['entry_foot_clearance_m'][:] = gap
  return ((margin - gap).clamp_min(0.) / .05).square().clamp_max(25.) * c.penalty_mask
