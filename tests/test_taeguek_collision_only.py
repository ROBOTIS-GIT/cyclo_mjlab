"""New references retain real-time timing and cannot mix checkpoint identities."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import source.tasks
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from source.tasks.mimic.motion_curriculum import MOTIONS, TAEGUEK_COLLISION_ONLY_NAMES, motion_identity, validate_checkpoint_motion
from source.tasks.mimic.koryo_schedule import initial_state
from scripts.reinforcement_learning.train_taeguek_contact_max import build_config
from scripts.reinforcement_learning.evaluate_koryo import evaluation_window


class TaegeukCollisionOnlyTests(unittest.TestCase):
  def test_all_chapters_match_koryo_contact_settings_and_real_time(self):
    koryo = load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Max')
    for i, name in enumerate(TAEGUEK_COLLISION_ONLY_NAMES, 1):
      with self.subTest(chapter=i):
        task, cfg = build_config(i)
        spec = MOTIONS[name]
        self.assertEqual(cfg.env.rewards, koryo.rewards)
        self.assertEqual(cfg.env.events, koryo.events)
        self.assertEqual(cfg.env.observations, koryo.observations)
        self.assertEqual(cfg.env.commands['reference_trajectory'].trajectory_file, str(spec.trajectory))
        self.assertEqual(load_runner_cls(task).motion_name, name)
        self.assertEqual(load_runner_cls(spec.training_task).motion_name, name)
        self.assertFalse(cfg.agent.resume)
        self.assertIsNone(cfg.checkpoint)
        identity = motion_identity(name)
        self.assertEqual(identity['fps'], 1/(cfg.env.decimation*cfg.env.sim.mujoco.timestep))
        self.assertGreater(identity['duration_s'], cfg.env.episode_length_s)
        _, horizon = evaluation_window(identity, True, 32, None, 30)
        self.assertGreater(horizon, identity['duration_s'])
        self.assertLess(horizon-identity['duration_s'], .15)
        source = spec.trajectory.with_name(f'k1_taeguek{i:02d}_collision_only_converted.npz')
        with np.load(source) as original:
          original_duration = (len(original['joint_pos'])-1)/float(original['fps'].item())
        self.assertLess(abs(identity['duration_s']-original_duration), .02)

  def test_chapter_state_is_saved_separately_and_rejects_other_chapters(self):
    for i, name in enumerate(TAEGUEK_COLLISION_ONLY_NAMES, 1):
      with self.subTest(chapter=i):
        task, _ = build_config(i)
        runner = object.__new__(load_runner_cls(task))
        cmd = SimpleNamespace(cfg=SimpleNamespace(trajectory_file=str(MOTIONS[name].trajectory)),
          curriculum_state_dict=lambda: initial_state(8), failure_score_by_bin=torch.zeros(3),
          load_curriculum_state_dict=Mock())
        runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=48,
          command_manager=SimpleNamespace(get_term=lambda _:cmd)))
        with patch.object(MjlabOnPolicyRunner, 'save') as save:
          runner.save('test.pt')
        infos = save.call_args.args[1]
        self.assertEqual(infos['motion_identity']['name'], name)
        self.assertEqual(infos[name+'_performance']['fixed_disturbance_stage'], 8)
        runner._restore_difficulty(infos)
        cmd.load_curriculum_state_dict.assert_called_once_with(initial_state(8))
        other = TAEGUEK_COLLISION_ONLY_NAMES[i % 8]
        with self.assertRaises(ValueError):
          validate_checkpoint_motion(infos, other)
        with patch('torch.load', return_value={'infos':{'motion_identity':motion_identity(other)}}), \
             patch.object(MjlabOnPolicyRunner, 'load') as load:
          with self.assertRaises(ValueError):
            runner.load('wrong.pt')
          load.assert_not_called()

  def test_legacy_taeguek_references_are_preserved(self):
    for chapter in [1,2]:
      legacy = MOTIONS[f'taeguek{chapter:02d}']
      self.assertEqual(legacy.trajectory.name, f'k1_taeguek{chapter:02d}_converted.npz')
      with self.assertRaises(ValueError):
        validate_checkpoint_motion({'motion_identity':motion_identity(legacy.name)},
          TAEGUEK_COLLISION_ONLY_NAMES[chapter-1])

  def test_selector_rejects_invalid_chapter_and_too_few_environments(self):
    for chapter in [0,9]:
      with self.assertRaises(ValueError):
        build_config(chapter)
    with self.assertRaises(ValueError):
      build_config(1, num_envs=1)
