"""Contact curriculum resumes PPO state and hands evaluated stages to workers."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from source.tasks.mimic.koryo_schedule import initial_state
from source.tasks.mimic.performance_runner import KoryoContactCurriculumRunner

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/reinforcement_learning/train_koryo_curriculum.py'
spec = importlib.util.spec_from_file_location('contact_driver', SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ContactCurriculumTests(unittest.TestCase):
  def test_task_keeps_contact_rewards_and_policy_inputs(self):
    fixed = load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo-ContactV2')
    task = 'Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Curriculum'
    curriculum = load_env_cfg(task)
    self.assertEqual(fixed.rewards, curriculum.rewards)
    self.assertEqual(fixed.observations, curriculum.observations)
    self.assertIn('koryo_difficulty', curriculum.curriculum)
    self.assertNotIn('performance_fixed_difficulty', curriculum.curriculum)
    self.assertIs(load_runner_cls(task), KoryoContactCurriculumRunner)

  def test_runner_restores_stage_before_reset_and_retains_ramp_optimizer_sampler(self):
    runner = object.__new__(KoryoContactCurriculumRunner)
    command = SimpleNamespace(failure_score_by_bin=torch.zeros(3), device='cpu',
                              frame_ids=torch.ones(2, dtype=torch.long), _update_command=Mock())
    command.load_curriculum_state_dict = Mock()
    env = SimpleNamespace(common_step_counter=0,
                          command_manager=SimpleNamespace(get_term=lambda _: command))
    def reset():
      self.assertEqual(command.load_curriculum_state_dict.call_args.args[0]['difficulty_stage'], 2)
    runner.env = SimpleNamespace(unwrapped=env, reset=Mock(side_effect=reset))
    runner.alg = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{'lr': 1e-5}]),
                                 learning_rate=.001)
    infos = {'koryo_performance': {'version': 2}, 'contact_training_steps': 48024,
             'koryo_curriculum': initial_state(2), 'performance_start_sampler': torch.arange(3.)}
    def base_load(*args):
      self.assertIsNone(args[1])  # Full PPO/optimizer load, not actor-only warm start.
      runner.current_learning_iteration = 2000
      return infos
    with patch('torch.load', return_value={'infos': infos}), \
         patch.object(MjlabOnPolicyRunner, 'load', side_effect=base_load):
      runner.load('checkpoint.pt')
    self.assertEqual(runner.current_learning_iteration, 2001)
    self.assertEqual(env.common_step_counter, 48024)
    self.assertEqual(runner.alg.learning_rate, 1e-5)
    torch.testing.assert_close(command.failure_score_by_bin, torch.arange(3.))
    runner.env.reset.assert_called_once()
    command._update_command.assert_called_once()

    command.curriculum_state_dict = lambda: initial_state(2)
    command.cfg = SimpleNamespace(trajectory_file=None)
    with patch.object(MjlabOnPolicyRunner, 'save') as save:
      runner.save('saved.pt')
    saved = save.call_args.args[1]
    self.assertEqual(saved['contact_training_steps'], 48024)
    self.assertEqual(saved['koryo_curriculum']['push_scale'], .5)
    self.assertTrue(saved['koryo_performance']['disturbance_curriculum'])
    self.assertNotIn('fixed_noise_scale', saved['koryo_performance'])

  def test_contact_driver_promotes_then_resumes_with_same_rewards_and_counters(self):
    stages = []
    def train(command, log, tee=False):
      self.assertIn('--contact-v2', command)
      checkpoint = torch.load(command[command.index('--checkpoint')+1], weights_only=False)
      stages.append(checkpoint['infos']['koryo_curriculum']['difficulty_stage'])
      checkpoint['iter'] += 1
      checkpoint['infos']['contact_training_steps'] += 24
      folder = Path(command[command.index('--output-dir')+1])
      torch.save(checkpoint, folder / f"model_{checkpoint['iter']}.pt")
    def evaluate(checkpoint, directory, name, iteration, noise, push, trials, device, motion):
      return [{'seed': iteration*3+i, 'condition': 'train', 'episodes': trials,
               'reference_end_successes': trials, 'full_motion_from_zero': True,
               'deterministic': True, 'noise_scale': noise, 'push_scale': push,
               'mean_body_error_m': .03, 'mean_anchor_error_m': .1} for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
      folder = Path(tmp)
      source = folder / 'input.pt'
      torch.save({'iter': 2000, 'infos': {'koryo_performance': {'version': 2},
                 'contact_training_steps': 48024}}, source)
      for index, checkpoint in enumerate([source, folder / 'run0/resume.pt']):
        output = folder / f'run{index}'
        argv = [str(SCRIPT), '--contact-v2', '--checkpoint', str(checkpoint),
                '--output-dir', str(output), '--additional-iterations', '2', '--eval-every', '1']
        with patch('sys.argv', argv), patch.object(driver, 'run_logged', side_effect=train), \
             patch.object(driver, 'evaluate', side_effect=evaluate):
          driver.main()
      result = torch.load(folder / 'run1/resume.pt', weights_only=False)
      self.assertEqual(stages, [0, 0, 1, 1])
      self.assertEqual(result['iter'], 2004)
      self.assertEqual(result['infos']['contact_training_steps'], 48120)
      self.assertEqual(result['infos']['koryo_curriculum']['difficulty_stage'], 2)
      self.assertEqual(torch.load(source, weights_only=False)['iter'], 2000)
      status = json.loads((folder / 'run1/status.json').read_text())
      self.assertEqual(status['curriculum']['push_scale'], .5)

  def test_rejects_wrong_reward_version_and_missing_ramp(self):
    for infos in ({}, {'koryo_performance': {'version': 1}},
                  {'koryo_performance': {'version': 2}}):
      with self.assertRaises(ValueError):
        driver.validate_contact_checkpoint(infos)


if __name__ == '__main__':
  unittest.main()
