"""Guard motion identity, evaluation horizons, and Taeguek curriculum wiring."""
import importlib.util
from pathlib import Path
import unittest

from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from source.tasks.mimic.curriculum_runner import Taeguek01CurriculumRunner
from source.tasks.mimic.mdp.koryo_curriculum import KoryoCurriculumCommandCfg, KoryoObservationNoise
from source.tasks.mimic.motion_curriculum import MOTIONS, motion_identity, validate_checkpoint_motion

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reinforcement_learning/evaluate_koryo.py"
spec = importlib.util.spec_from_file_location("motion_evaluator", SCRIPT)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class TaeguekTests(unittest.TestCase):
  def test_task_keeps_taeguek_motion_with_shared_curriculum(self):
    task = MOTIONS["taeguek01"]
    cfg = load_env_cfg(task.training_task)
    self.assertEqual(Path(cfg.commands["reference_trajectory"].trajectory_file), task.trajectory)
    self.assertIsInstance(cfg.commands["reference_trajectory"], KoryoCurriculumCommandCfg)
    self.assertIn("taeguek01_difficulty", cfg.curriculum)
    self.assertNotIn("koryo_difficulty", cfg.curriculum)
    self.assertIs(load_runner_cls(task.training_task), Taeguek01CurriculumRunner)
    self.assertEqual(cfg.episode_length_s, 30.0)
    self.assertEqual(cfg.events["physics_material"].mode, "reset")
    self.assertIsInstance(cfg.observations["actor"].terms["base_ang_vel"].noise, KoryoObservationNoise)

  def test_play_and_baseline_remain_without_curriculum(self):
    task = MOTIONS["taeguek01"]
    baseline = load_env_cfg(task.task)
    play = load_env_cfg(task.training_task, play=True)
    self.assertNotIsInstance(baseline.commands["reference_trajectory"], KoryoCurriculumCommandCfg)
    self.assertTrue(play.commands["reference_trajectory"].start_from_zero)
    self.assertNotIn("taeguek01_difficulty", play.curriculum)
    self.assertNotIn("push_robot", play.events)

  def test_full_motion_horizon_is_31_seconds_not_koryos_85_seconds(self):
    identity = motion_identity("taeguek01")
    self.assertEqual(identity["frames"], 1582)
    self.assertEqual(identity["fps"], 50.)
    self.assertAlmostEqual(identity["duration_s"], 31.62)
    starts, cap = evaluator.evaluation_window(identity, True, 32, None, 30.)
    self.assertEqual(starts, [0.] * 32)
    self.assertGreater(cap, identity["duration_s"])
    self.assertLess(cap, 32.)
    starts, cap = evaluator.evaluation_window(identity, False, 32, None, 30.)
    self.assertEqual(starts, list(range(32)))
    with self.assertRaises(ValueError):
      evaluator.evaluation_window(identity, False, 32, [32.], 30.)

  def test_koryo_legacy_compatibility_but_no_cross_motion_resume(self):
    validate_checkpoint_motion({"koryo_curriculum": {}}, "koryo")
    with self.assertRaises(ValueError):
      validate_checkpoint_motion({"koryo_curriculum": {}}, "taeguek01")
    with self.assertRaises(ValueError):
      validate_checkpoint_motion({"motion_identity": motion_identity("koryo")}, "taeguek01")
    with self.assertRaises(ValueError):
      validate_checkpoint_motion({"motion_identity": motion_identity("taeguek01")}, "koryo")

  def test_changed_motion_contents_cannot_reuse_old_gate_state(self):
    identity = motion_identity("taeguek01")
    validate_checkpoint_motion({"motion_identity": identity}, "taeguek01")
    changed = {**identity, "sha256": "different motion"}
    with self.assertRaises(ValueError):
      validate_checkpoint_motion({"motion_identity": changed}, "taeguek01")


if __name__ == "__main__":
  unittest.main()
