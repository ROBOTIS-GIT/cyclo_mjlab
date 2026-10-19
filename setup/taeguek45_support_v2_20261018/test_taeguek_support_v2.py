"""Reward-behavior and task-isolation regression tests."""
from unittest import TestCase
import torch
import source.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from source.tasks.mimic.mdp.taeguek_support_v2 import (
  clearance_cost, kick_height_cost, sole_bottom_height)


class GeometryTests(TestCase):
  def test_rotated_lowest_sphere_not_ankle_height(self):
    p = torch.tensor([[[0., 0., 1.]]])
    q = torch.tensor([[[1., 0., 0., 0.]]])
    offsets = torch.tensor([[[0., 0., -.1], [.2, 0., 0.]]])
    radii = torch.tensor([[.02, .03]])
    self.assertAlmostEqual(sole_bottom_height(p,q,offsets,radii).item(), .88, places=6)
    # 180 degrees around Y: the second sphere becomes the lowest surface.
    q = torch.tensor([[[0., 0., 1., 0.]]])
    self.assertAlmostEqual(sole_bottom_height(p,q,offsets,radii).item(), .97, places=6)

  def test_clearance_tolerance_monotonicity_and_recovery_bound(self):
    actual = torch.tensor([[.001, .003, .008, .016, .5]])
    ref = torch.zeros_like(actual)
    cost = clearance_cost(actual, ref, torch.ones_like(actual,dtype=torch.bool))
    self.assertEqual(cost[0,0],0); self.assertEqual(cost[0,1],0)
    self.assertGreater(cost[0,3],cost[0,2]); self.assertEqual(cost[0,4],4)

  def test_intended_kick_and_reference_clearance_not_pinned(self):
    actual = torch.tensor([[.012, 1.2]])
    reference = torch.tensor([[.012, 1.2]])
    support = torch.tensor([[True,False]])
    self.assertEqual(clearance_cost(actual,reference,support).sum(),0)
    actual[0,1] = 1.6
    self.assertEqual(clearance_cost(actual,reference,support).sum(),0)

  def test_kick_lowering_overshoot_and_body_hop_have_cost(self):
    ref = torch.tensor([[1.2,0.]])
    pelvis = torch.tensor([.8]); kick=torch.tensor([[True,False]])
    correct=kick_height_cost(ref,ref,pelvis,pelvis,kick)
    low=kick_height_cost(ref-.06,ref,pelvis,pelvis,kick)
    high=kick_height_cost(ref+.06,ref,pelvis,pelvis,kick)
    hop=kick_height_cost(ref,ref,pelvis+.06,pelvis,kick)
    self.assertEqual(correct.sum(),0)
    self.assertGreater(low[0,0],0);self.assertEqual(low[0,1],0)
    torch.testing.assert_close(low,high)
    self.assertGreater(hop[0,0],0)
    # Suppressing the kick does not deactivate reference-based eligibility.
    absent=kick_height_cost(torch.zeros_like(ref),ref,pelvis,pelvis,kick)
    self.assertEqual(absent[0,0],4)


class ConfigTests(TestCase):
  def test_v2_changes_only_support_reward(self):
    for chapter in (4,5):
      base=load_env_cfg(f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-SupportV1')
      new=load_env_cfg(f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-SupportV2')
      for name in ['observations','actions','events','terminations','commands','sim']:
        self.assertEqual(getattr(base,name),getattr(new,name),name)
      # Robot scene factories contain newly constructed function objects, so
      # dataclass identity equality is inappropriate for the entire scene.
      self.assertEqual(base.scene.sensors,new.scene.sensors)
      self.assertEqual(base.scene.num_envs,new.scene.num_envs)
      self.assertEqual(base.scene.terrain,new.scene.terrain)
      self.assertEqual(set(base.rewards),set(new.rewards))
      for name in base.rewards:
        if name != 'taeguek_support_quality':self.assertEqual(base.rewards[name],new.rewards[name])

  def test_v1_all_chapters_unchanged(self):
    for chapter in range(2,9):
      cfg=load_env_cfg(f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-SupportV1')
      term=cfg.rewards['taeguek_support_quality']
      self.assertEqual(term.params['support_weight'],.45)
      self.assertEqual(term.params['ramp_steps'],24000)
      self.assertEqual(term.func.__name__,'TaegeukFootSupportQuality')

  def test_finetune_uses_fixed_small_rate(self):
    cfg=load_rl_cfg('Cyclo-Mimic-K1-Rev1-Taeguek04-SupportV2')
    self.assertEqual(cfg.algorithm.learning_rate,1e-4)
    self.assertEqual(cfg.algorithm.schedule,'fixed')
