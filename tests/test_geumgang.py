"""Geumgang conversion, task parity and checkpoint isolation."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import source.tasks
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from source.tasks.mimic.koryo_schedule import initial_state
from source.tasks.mimic.motion_curriculum import MOTIONS, motion_identity, validate_checkpoint_motion
from source.tasks.mimic.performance_runner import GeumgangContactMaxRunner
from scripts.reinforcement_learning.evaluate_koryo import evaluation_window

TASK = 'Cyclo-Mimic-K1-Rev1-Geumgang-ContactV2-Max'


class GeumgangTests(unittest.TestCase):
  def test_task_matches_koryo_except_motion_and_run_name(self):
    cfg = load_env_cfg(TASK)
    koryo = load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Max')
    self.assertEqual(cfg.rewards, koryo.rewards)
    self.assertEqual(cfg.observations, koryo.observations)
    self.assertEqual(cfg.events, koryo.events)
    self.assertEqual(cfg.commands['reference_trajectory'].trajectory_file, str(MOTIONS['geumgang'].trajectory))
    self.assertEqual(load_rl_cfg(TASK).algorithm, load_rl_cfg('Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Max').algorithm)
    self.assertIs(load_runner_cls(TASK), GeumgangContactMaxRunner)
    self.assertEqual(load_runner_cls(MOTIONS['geumgang'].training_task).motion_name, 'geumgang')

  def test_converted_clip_and_evaluation_cover_full_geumgang(self):
    identity = motion_identity('geumgang')
    self.assertEqual(identity['frames'], 4994)
    self.assertEqual(identity['fps'], 50)
    self.assertAlmostEqual(identity['duration_s'], 99.86)
    starts, cap = evaluation_window(identity, True, 32, None, 30)
    self.assertEqual(starts, [0.]*32)
    self.assertGreater(cap, 99.86)
    with np.load(MOTIONS['geumgang'].trajectory) as motion:
      self.assertEqual(motion['joint_pos'].shape, (4994,23))
      for key in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
        self.assertTrue(np.isfinite(motion[key]).all(), key)
      np.testing.assert_allclose(np.linalg.norm(motion['body_quat_w'],axis=-1),1,atol=1e-5)

  def test_save_and_restore_use_geumgang_identity_and_state(self):
    runner = object.__new__(GeumgangContactMaxRunner)
    cmd = SimpleNamespace(cfg=SimpleNamespace(trajectory_file=str(MOTIONS['geumgang'].trajectory)),
      curriculum_state_dict=lambda: initial_state(8), failure_score_by_bin=torch.zeros(3),
      load_curriculum_state_dict=Mock())
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=48,
      command_manager=SimpleNamespace(get_term=lambda _: cmd)))
    with patch.object(MjlabOnPolicyRunner, 'save') as save:
      runner.save('geumgang.pt')
    infos = save.call_args.args[1]
    self.assertEqual(infos['motion_identity'], motion_identity('geumgang'))
    self.assertIn('geumgang_performance', infos)
    self.assertIn('geumgang_curriculum', infos)
    self.assertNotIn('koryo_performance', infos)
    self.assertNotIn('koryo_curriculum', infos)
    runner._restore_difficulty(infos)
    cmd.load_curriculum_state_dict.assert_called_once_with(initial_state(8))
    with self.assertRaises(ValueError):
      validate_checkpoint_motion(infos, 'koryo')

  def test_rejects_other_motion_before_loading_networks(self):
    runner = object.__new__(GeumgangContactMaxRunner)
    for infos in [{}, {'motion_identity': motion_identity('koryo')}]:
      with patch('torch.load', return_value={'infos':infos}), patch.object(MjlabOnPolicyRunner, 'load') as load:
        with self.assertRaises(ValueError):
          runner.load('wrong.pt')
        load.assert_not_called()
