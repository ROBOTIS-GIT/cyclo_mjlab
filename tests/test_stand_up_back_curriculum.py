"""Regression checks for preparation coverage and pose-aware velocity rewards."""
from types import SimpleNamespace as S
from unittest.mock import patch
import math
import unittest

import torch

from source.tasks.mimic.config.k1_rev1.umr_stand_up_back_env_cfg import (
  k1_rev1_umr_stand_up_back_env_cfg,
)
from source.tasks.mimic.mdp.stand_up_curriculum import (
  StandUpCurriculumCommand,
  StandUpCurriculumCommandCfg,
  curriculum_push,
)
from source.tasks.mimic.mdp.pose_gated_rewards import (
  pose_gated_linear_velocity_tracking,
  pose_gated_angular_velocity_tracking,
)


class StandUpBackCurriculum(unittest.TestCase):
  def test_failure_peak_cannot_starve_preparation_or_whole_clip(self):
    cfg = k1_rev1_umr_stand_up_back_env_cfg().commands["reference_trajectory"]
    self.assertIsInstance(cfg, StandUpCurriculumCommandCfg)
    n = 8192
    c = object.__new__(StandUpCurriculumCommand)
    c.cfg = cfg
    c._env = S(num_envs=n, device="cpu", step_dt=0.02,
               termination_manager=S(terminated=torch.zeros(n, dtype=torch.bool)))
    c.reference = S(num_frames=726)
    c.frame_ids = torch.zeros(n, dtype=torch.long)
    c.start_bin_count = 15
    c.failure_score_by_bin = torch.zeros(15)
    c.failure_score_by_bin[10] = 1e6
    c._episode_failures_by_bin = torch.zeros(15)
    c.failure_smoothing_kernel = torch.ones(cfg.failure_sampling_kernel_size)
    c.failure_smoothing_kernel /= c.failure_smoothing_kernel.sum()
    c.metrics = {key: torch.zeros(n) for key in (
      "start_sampling_entropy", "start_sampling_top1_prob",
      "start_sampling_top1_bin", "preparation_start_fraction",
    )}
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(123)
      c._sample_start_frames(torch.arange(n))
    fixed = int(n * 0.3)
    self.assertTrue((c.frame_ids[:fixed] == 0).all())
    times = c.frame_ids[fixed:] * 0.02
    # Even with all failures in the old 9.7 s bin, preparation has guaranteed
    # coverage and earlier/later parts still receive uniform exploration.
    preparation = (times >= 7.5) & (times <= 9.5)
    self.assertGreater(float(preparation.float().mean()), 0.35)
    outside = (times < 7.5) | (times > 10.7)
    self.assertGreater(float(outside.float().mean()), 0.15)
    self.assertLess(float(c.metrics["start_sampling_top1_prob"][0]), 0.54)

  def test_wrong_pose_does_not_earn_full_velocity_reward(self):
    zeros = torch.zeros(2, 1, 3)
    reference_q = torch.tensor([[[1., 0., 0., 0.]]] * 2)
    actual_q = reference_q.clone()
    actual_q[1, 0] = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0., 0.])
    c = S(cfg=S(body_names=("torso",)),
          aligned_body_position_w=zeros, robot_body_pos_w=zeros,
          aligned_body_orientation_w=reference_q, robot_body_quat_w=actual_q,
          body_lin_vel_w=zeros, robot_body_lin_vel_w=zeros,
          body_ang_vel_w=zeros, robot_body_ang_vel_w=zeros)
    env = S(command_manager=S(get_term=lambda _: c))
    for reward in (pose_gated_linear_velocity_tracking, pose_gated_angular_velocity_tracking):
      result = reward(env, "reference_trajectory", std=1.)
      self.assertAlmostEqual(float(result[0]), 1.)
      self.assertLess(float(result[1]), 1e-5)

  def test_pushes_start_disabled_and_reach_original_range(self):
    c = S(difficulty_stage=0)
    env = S(command_manager=S(get_term=lambda _: c))
    ranges = {"x": (-0.5, 0.5)}
    with patch("source.tasks.mimic.mdp.stand_up_curriculum.push_by_setting_velocity") as push:
      curriculum_push(env, None, ranges)
      push.assert_not_called()
      c.difficulty_stage = 3
      curriculum_push(env, None, ranges)
      self.assertEqual(push.call_args.kwargs["velocity_range"], ranges)

  def test_play_starts_at_zero_without_training_curriculum(self):
    cfg = k1_rev1_umr_stand_up_back_env_cfg(play=True)
    c = cfg.commands["reference_trajectory"]
    self.assertTrue(c.start_from_zero)
    self.assertNotIsInstance(c, StandUpCurriculumCommandCfg)
    self.assertNotIn("push_robot", cfg.events)


if __name__ == "__main__":
  unittest.main()
