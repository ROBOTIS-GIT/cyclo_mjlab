"""Full disturbances must apply on the first reset and survive checkpoint resume."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
import source.tasks
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from source.tasks.mimic.koryo_schedule import initial_state
from source.tasks.mimic.mdp.koryo_curriculum import KoryoCurriculumCommandCfg, KoryoObservationNoise
from source.tasks.mimic.performance_runner import KoryoContactMaxRunner

TASK = 'Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Max'


class ContactMaxTests(TestCase):
  def test_full_stage_is_set_during_build_before_first_reset(self):
    cfg = load_env_cfg(TASK)
    cmd = Mock()
    with patch.object(KoryoCurriculumCommandCfg, 'build', return_value=cmd):
      built = cfg.commands['reference_trajectory'].build(Mock())
    self.assertIs(built, cmd)
    cmd.load_curriculum_state_dict.assert_called_once_with(initial_state(8))
    noises = [t.noise for t in cfg.observations['actor'].terms.values()
              if isinstance(t.noise, KoryoObservationNoise)]
    self.assertTrue(noises)
    self.assertTrue(all(n.curriculum_scale == 1 for n in noises))
    self.assertEqual(cfg.rewards['action_rate_l2'].weight, -.1)
    self.assertIn('performance_contact_transitions', cfg.rewards)
    self.assertIs(load_runner_cls(TASK), KoryoContactMaxRunner)
    rl = load_rl_cfg(TASK)
    self.assertEqual((rl.algorithm.learning_rate, rl.algorithm.schedule,
                      rl.algorithm.entropy_coef), (.001, 'adaptive', .005))

  def test_checkpoint_records_and_restores_full_stage_and_ramp(self):
    runner = object.__new__(KoryoContactMaxRunner)
    cmd = SimpleNamespace(cfg=SimpleNamespace(trajectory_file=None),
                          curriculum_state_dict=lambda: initial_state(8),
                          failure_score_by_bin=torch.zeros(3),
                          load_curriculum_state_dict=Mock())
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(
      common_step_counter=24024, command_manager=SimpleNamespace(get_term=lambda _: cmd)))
    with patch.object(MjlabOnPolicyRunner, 'save') as save:
      runner.save('test.pt')
    infos = save.call_args.args[1]
    self.assertEqual(infos['contact_training_steps'], 24024)
    self.assertEqual(infos['koryo_performance']['fixed_noise_scale'], 1)
    self.assertEqual(infos['koryo_performance']['push_scale'], 1)
    self.assertTrue(runner.resumes_performance_state(infos['koryo_performance']))
    runner._restore_difficulty(infos)
    cmd.load_curriculum_state_dict.assert_called_once_with(initial_state(8))
    with self.assertRaises(ValueError):
      runner.resumes_performance_state({'version': 2})
    with self.assertRaises(ValueError):
      runner._restore_difficulty({'koryo_curriculum': initial_state(0)})
