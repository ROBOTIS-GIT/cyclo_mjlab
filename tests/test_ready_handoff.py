"""Handoff timing, stability gating, gains, and history regression tests."""
from types import SimpleNamespace as S
from unittest.mock import patch
import unittest
import torch

from source.tasks.mimic.mdp.ready_handoff import (
  ReadyHandoffCommand as Command, ReadyHandoffCommandCfg,
  ReadyHandoffAction as Action, handoff_torque_rate,
)
from source.tasks.mimic.mdp.commands import ReferenceTrajectoryCommand
from source.tasks.mimic.config.k1_rev1.umr_stand_up_ready_handoff_env_cfg import (
  k1_rev1_umr_stand_up_ready_handoff_env_cfg,
)


def command():
  c = object.__new__(Command)
  c.cfg = ReadyHandoffCommandCfg(asset_name='robot', anchor_body_name='torso', body_names=('torso',),
                                resampling_time_range=(1e9, 1e9), trajectory_file='unused.npz')
  c._env = S(num_envs=2, device='cpu', step_dt=.02)
  c.reference = S(joint_position=torch.zeros(10, 2))
  c.robot_anchor_body_id = 0
  c.robot = S(data=S(joint_pos=torch.zeros(2, 2), joint_vel=torch.zeros(2, 2),
                    body_link_ang_vel_w=torch.zeros(2, 1, 3)))
  c.enabled = torch.tensor([True, False])
  c.phase = torch.tensor([0, 2])
  c.freeze_reference = c.enabled.clone()
  c.penalty_mask = c.enabled.clone()
  c.timeout = torch.zeros(2, dtype=torch.bool)
  for key in ('ready_elapsed', 'stable_elapsed', 'blend_elapsed', 'motion_elapsed'):
    setattr(c, key, torch.zeros(2))
  c.hold_duration = torch.full((2,), .4)
  c.blend_duration = torch.full((2,), .8)
  c.metrics = {key: torch.zeros(2) for key in ('handoff_alpha', 'handoff_phase', 'handoff_ready_timeout')}
  return c


class Handoff(unittest.TestCase):
  def test_hold_blend_endpoints_and_intermediate_start_bypass(self):
    c = command()
    alphas, freeze = [], []
    for _ in range(80):
      alpha = c.advance_handoff()
      alphas.append(float(alpha[0])); freeze.append(bool(c.freeze_reference[0]))
      self.assertEqual(float(alpha[1]), 1.)
      self.assertFalse(c.freeze_reference[1])
    self.assertEqual(alphas[0], 0.)
    self.assertEqual(alphas[-1], 1.)
    self.assertTrue(all(b >= a for a, b in zip(alphas, alphas[1:])))
    first_full = alphas.index(1.)
    self.assertTrue(freeze[first_full])
    self.assertFalse(freeze[first_full+1])

  def test_unstable_ready_cannot_enter_and_times_out(self):
    for kind in ('position', 'velocity', 'body_velocity'):
      c = command()
      if kind == 'position': c.robot.data.joint_pos[0] = .3
      if kind == 'velocity': c.robot.data.joint_vel[0] = 1.
      if kind == 'body_velocity': c.robot.data.body_link_ang_vel_w[0] = 1.
      for _ in range(155):
        self.assertEqual(float(c.advance_handoff()[0]), 0.)
      self.assertTrue(c.timeout[0])
      self.assertFalse(c.timeout[1])

  def test_reference_clock_freezes_only_selected_envs(self):
    c = command(); c.frame_ids = torch.tensor([0, 5])
    def update(obj): obj.frame_ids += 1
    with patch.object(ReferenceTrajectoryCommand, '_update_command', update):
      c._update_command()
    self.assertEqual(c.frame_ids.tolist(), [0, 6])
    velocity = torch.ones(2, 3, 3)
    out = c._held_velocity(velocity)
    self.assertEqual(float(out[0].sum()), 0.)
    self.assertEqual(float(out[1].sum()), 9.)

  def test_applied_targets_and_gains_interpolate_together(self):
    a = object.__new__(Action)
    a._env = S(sim=S(model=S(actuator_gainprm=torch.zeros(3, 1, 3), actuator_biasprm=torch.zeros(3, 1, 3))))
    a.command = S(advance_handoff=lambda: torch.tensor([0., .5, 1.]))
    a._raw_actions = torch.zeros(3, 1)
    a._scale = 1.; a._offset = 0.
    a.ready_target = torch.tensor([.36])
    a.applied_target = a.ready_target.repeat(3, 1)
    a.previous_target = a.applied_target.clone()
    a.ready_kp = torch.tensor([150.]); a.policy_kp = torch.tensor([76.45])
    a.ready_kd = torch.tensor([10.]); a.policy_kd = torch.tensor([1.42])
    a.ctrl_ids = torch.tensor([0])
    a.process_actions(torch.full((3, 1), .56))
    torch.testing.assert_close(a.applied_target[:, 0], torch.tensor([.36, .46, .56]))
    torch.testing.assert_close(a._env.sim.model.actuator_gainprm[:, 0, 0], torch.tensor([150., 113.225, 76.45]))
    torch.testing.assert_close(a._env.sim.model.actuator_biasprm[:, 0, 2], -torch.tensor([10., 5.71, 1.42]))

  def test_reset_torque_is_not_compared_to_previous_episode(self):
    c = S(penalty_mask=torch.tensor([True]), metrics={k:torch.zeros(1) for k in ('handoff_torque_delta_nm','handoff_peak_torque_delta_nm')})
    a = S(command=c, ctrl_ids=torch.tensor([0]), previous_torque=torch.tensor([[100.]]), torque_valid=torch.tensor([False]))
    env = S(action_manager=S(get_term=lambda _:a), sim=S(data=S(actuator_force=torch.tensor([[10.]]))))
    self.assertEqual(float(handoff_torque_rate(env)), 0.)
    env.sim.data.actuator_force[:] = 20.
    self.assertEqual(float(handoff_torque_rate(env)), 1.)

  def test_play_and_training_include_same_controller(self):
    train = k1_rev1_umr_stand_up_ready_handoff_env_cfg()
    play = k1_rev1_umr_stand_up_ready_handoff_env_cfg(play=True)
    self.assertEqual(train.commands['reference_trajectory'].first_frame_env_fraction, .3)
    self.assertEqual(play.commands['reference_trajectory'].first_frame_env_fraction, 1.)
    self.assertEqual(play.commands['reference_trajectory'].handoff_seconds, (.8, .8))
    self.assertEqual(train.actions['joint_pos'].ready_kp, play.actions['joint_pos'].ready_kp)
    self.assertIn('ready_timeout', train.terminations)


if __name__ == '__main__': unittest.main()
