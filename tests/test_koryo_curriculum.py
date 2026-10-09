"""Check curriculum accounting, noise scaling and checkpoint compatibility."""
from types import SimpleNamespace as S
from unittest.mock import Mock, patch
import unittest
import torch

from source.tasks.mimic.mdp.koryo_curriculum import (
  KoryoCurriculumCommand as Command, KoryoObservationNoise, koryo_material,
)
from source.tasks.mimic.curriculum_runner import KoryoCurriculumRunner as Runner
from mjlab.rl import MjlabOnPolicyRunner


def command():
  c = object.__new__(Command)
  c.cfg = S(first_frame_env_fraction=0.5, curriculum_min_attempts=4, curriculum_success_threshold=0.75)
  c._env = S(num_envs=4, device='cpu', max_episode_length=1500,
             episode_length_buf=torch.tensor([1500, 1500, 1500, 1500]),
             termination_manager=S(terminated=torch.zeros(4, dtype=torch.bool)))
  c.difficulty_stage = 0
  c.attempts = c.successes = 0
  c.last_success_rate = 0.
  c.trial_stage = torch.zeros(4, dtype=torch.long)
  return c


class KoryoCurriculumTests(unittest.TestCase):
  def test_only_fixed_group_and_failure_overrides_timeout(self):
    c = command()
    c._env.termination_manager.terminated[1] = True
    c.record_outcomes(slice(None))
    self.assertEqual((c.attempts, c.successes), (2, 1))
    c.record_outcomes(slice(None))
    self.assertEqual(c.attempts, 2)

  def test_ignores_initial_interrupted_and_old_stage_attempts(self):
    c = command()
    c._env.episode_length_buf[:] = 5
    c.record_outcomes(slice(None))
    self.assertEqual(c.attempts, 0)
    c._env.episode_length_buf[:] = 1500
    c.difficulty_stage = 1
    c.trial_stage[:] = 0
    c.record_outcomes(slice(None))
    self.assertEqual(c.attempts, 0)

  def test_promotes_only_after_enough_successes(self):
    c = command()
    c.record_outcomes(slice(None))
    self.assertEqual(c.difficulty_stage, 0)
    c.trial_stage[:] = 0
    c._env.termination_manager.terminated[1] = True
    c.record_outcomes(slice(None))
    self.assertEqual(c.difficulty_stage, 1)
    self.assertEqual(c.last_success_rate, 0.75)
    self.assertEqual(c.attempts, 0)

  def test_noise_scale_changes_after_tensor_cache_is_populated(self):
    noise = KoryoObservationNoise(n_min=-1., n_max=1.)
    data = torch.ones(100)
    torch.manual_seed(4)
    low = noise.apply(data) - data
    noise.curriculum_scale = 1.
    torch.manual_seed(4)
    high = noise.apply(data) - data
    torch.testing.assert_close(high, low * 4)

  def test_friction_reaches_original_range(self):
    c = command()
    env = S(command_manager=S(get_term=lambda _: c))
    with patch('source.tasks.mimic.mdp.koryo_curriculum.randomize_rigid_body_material') as f:
      koryo_material(env, None, (.3, 1.2))
      self.assertAlmostEqual(f.call_args.kwargs['ranges'][0], .6375)
      c.difficulty_stage = 3
      koryo_material(env, None, (.3, 1.2))
      self.assertAlmostEqual(f.call_args.kwargs['ranges'][0], .3)
      self.assertAlmostEqual(f.call_args.kwargs['ranges'][1], 1.2)

  def test_checkpoint_roundtrip_and_actor_only(self):
    c = command(); c.difficulty_stage = 2; c.attempts = 50; c.successes = 40
    runner = object.__new__(Runner)
    runner.env = S(unwrapped=S(command_manager=S(get_term=lambda _: c)), reset=Mock())
    runner._refresh_reference = Mock()
    with patch.object(MjlabOnPolicyRunner, 'save') as save:
      runner.save('unused.pt'); infos = save.call_args.args[1]
    c.difficulty_stage = 0
    with patch.object(MjlabOnPolicyRunner, 'load', return_value=infos):
      runner.load('unused.pt', load_cfg={'actor': True})
      self.assertEqual(c.difficulty_stage, 0)
      runner.load('unused.pt')
    self.assertEqual((c.difficulty_stage, c.attempts, c.successes), (2, 50, 40))
    self.assertTrue(torch.all(c.trial_stage == -1))
    runner.env.reset.assert_called_once()

  def test_random_initial_clock_is_disabled(self):
    runner = object.__new__(Runner)
    with patch.object(MjlabOnPolicyRunner, 'learn') as learn:
      runner.learn(2, init_at_random_ep_len=True)
    learn.assert_called_once_with(2, init_at_random_ep_len=False)


if __name__ == '__main__':
  unittest.main()
