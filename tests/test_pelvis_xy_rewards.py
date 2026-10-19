"""CPU checks for world-path coordinates and all registered Taegeuk tasks."""
import math
import unittest
from types import SimpleNamespace

import torch

from source.tasks.mimic.mdp.commands import ReferenceTrajectoryCommand
from source.tasks.mimic.mdp.pelvis_xy_rewards import (
  reference_pelvis_xy,
  xy_position_score,
)


class PelvisXYTests(unittest.TestCase):
  def test_score_and_invalid_inputs(self):
    actual = torch.tensor([[0., 0., 10.], [.5, 0., -10.], [1., 0., 0.]])
    ref = torch.zeros_like(actual)
    expected = torch.tensor([1., math.exp(-1), math.exp(-4)])
    torch.testing.assert_close(xy_position_score(actual, ref), expected)
    origin = torch.tensor([10., -20., 5.])
    torch.testing.assert_close(xy_position_score(actual+origin, ref+origin), expected)
    for value in (float("nan"), float("inf"), -float("inf")):
      self.assertEqual(xy_position_score(torch.full((1, 3), value), ref[:1]).item(), 0.)
    for std in (0., -1., float("nan"), float("inf")):
      with self.assertRaises(ValueError):
        xy_position_score(actual, ref, std)

  def test_real_command_properties_and_pelvis_mapping(self):
    # Exercise the real reference/robot property implementations with two
    # different frames and origins. Pelvis is deliberately not body index 0.
    command = object.__new__(ReferenceTrajectoryCommand)
    command.cfg = SimpleNamespace(body_names=("torso_link", "pelvis"))
    command.frame_ids = torch.tensor([1, 0])
    command.reference = SimpleNamespace(body_position_w=torch.tensor([
      [[0., 0., 1.], [2., 3., 1.]],
      [[0., 0., 1.], [4., 6., 1.]],
    ]))
    command._env = SimpleNamespace(scene=SimpleNamespace(
      env_origins=torch.tensor([[10., 20., 0.], [-10., -20., 0.]])
    ))
    command.tracked_body_ids = [2, 0]
    bodies = torch.zeros(2, 3, 3)
    bodies[:, 0] = torch.tensor([[14.5, 26., 99.], [-8., -17., -99.]])
    command.robot = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=bodies))
    command.aligned_body_position_w = command.robot_body_pos_w.clone()
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    torch.testing.assert_close(reference_pelvis_xy(env), torch.tensor([math.exp(-1), 1.]))
    # Moving only the robot by another 0.5m must penalize accumulated drift.
    bodies[0, 0, 0] += .5
    self.assertAlmostEqual(reference_pelvis_xy(env)[0].item(), math.exp(-4), places=6)

  def test_task_registration_and_scope(self):
    import source.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from source.tasks.mimic.config.k1_rev1.base_env_cfg import k1_rev1_mimic_env_cfg

    tasks = [f"Cyclo-Mimic-K1-Rev1-Taeguek{i:02d}" for i in range(1, 9)]
    tasks += [f"Cyclo-Mimic-K1-Rev1-Taeguek{i:02d}-Curriculum" for i in (1, 2)]
    for task in tasks:
      for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        term = cfg.rewards["reference_pelvis_xy"]
        self.assertIs(term.func, reference_pelvis_xy)
        self.assertEqual(term.weight, 1.0)
        self.assertEqual(term.params, {"command_name": "reference_trajectory", "std": .5})
        self.assertIn("pelvis", cfg.commands["reference_trajectory"].body_names)
        base = k1_rev1_mimic_env_cfg(play=play)
        self.assertEqual(set(cfg.rewards), set(base.rewards) | {"reference_pelvis_xy"})
        for name, reward in base.rewards.items():
          self.assertEqual(cfg.rewards[name], reward)
    for task in ("Cyclo-Mimic-K1-Rev1-Koryo", "Cyclo-Mimic-K1-Rev1-Koryo-SupportV3-Max", "Cyclo-Mimic-K1-Rev1-Dance1"):
      self.assertNotIn("reference_pelvis_xy", load_env_cfg(task).rewards)


if __name__ == "__main__":
  unittest.main()
