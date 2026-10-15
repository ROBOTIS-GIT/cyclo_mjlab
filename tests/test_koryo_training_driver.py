"""Exercise block handoff, completion and evaluation failure without GPU work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from source.tasks.mimic.koryo_schedule import initial_state
from source.tasks.mimic.motion_curriculum import MOTIONS, motion_identity

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reinforcement_learning/train_koryo_curriculum.py"
spec = importlib.util.spec_from_file_location("koryo_driver", SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def fake_training(command, log, tee=False):
  count = int(command[command.index("--additional-iterations")+1])
  folder = Path(command[command.index("--output-dir")+1])
  motion = command[command.index("--motion")+1]
  if "--checkpoint" in command:
    saved = torch.load(Path(command[command.index("--checkpoint")+1]), weights_only=False)
  else:
    saved = {"iter": -1, "infos": {MOTIONS[motion].state_key: initial_state(),
                                  "motion_identity": motion_identity(motion)}}
  saved["iter"] += count
  torch.save(saved, folder / f"model_{saved['iter']}.pt")


def fake_evaluation(checkpoint, directory, name, iteration, noise, push, trials, device, motion="koryo"):
  return [{"motion": motion, "condition": "train", "seed": iteration*3+i, "episodes": trials,
           "reference_end_successes": 30, "full_motion_from_zero": True,
           "deterministic": True, "noise_scale": noise, "push_scale": push,
           "mean_body_error_m": .04, "mean_anchor_error_m": .5}
          for i in range(3)]


class DriverTests(unittest.TestCase):
  def run_case(self, folder, state, iterations=2, evaluation=fake_evaluation):
    source = folder / "original.pt"
    output = folder / "run"
    torch.save({"iter": 10, "infos": {"koryo_curriculum": state}}, source)
    argv = [str(SCRIPT), "--checkpoint", str(source), "--output-dir", str(output),
            "--additional-iterations", str(iterations), "--eval-every", "1", "--num-envs", "16"]
    with patch("sys.argv", argv), patch.object(driver, "run_logged", side_effect=fake_training), \
         patch.object(driver, "evaluate", side_effect=evaluation) as evaluator:
      driver.main()
    self.assertEqual(torch.load(source, weights_only=False)["iter"], 10)
    self.assertEqual(torch.load(source, weights_only=False)["infos"]["koryo_curriculum"], state)
    return output, evaluator.call_count

  def test_evaluated_state_is_handed_to_next_block_and_best_is_preserved(self):
    with tempfile.TemporaryDirectory() as tmp:
      folder, calls = self.run_case(Path(tmp), initial_state(3))
      status = json.loads((folder / "status.json").read_text())
      self.assertEqual(status["status"], "budget_exhausted")
      self.assertEqual(status["curriculum"]["push_scale"], .6)
      self.assertEqual(status["completed_additional_iterations"], 2)
      self.assertEqual(calls, 4)  # Current and full at each of two checkpoints.
      saved = torch.load(folder / "resume.pt", weights_only=False)
      self.assertEqual(saved["iter"], 12)
      self.assertEqual(saved["infos"]["koryo_curriculum"], status["curriculum"])
      self.assertEqual(torch.load(folder / "best_full.pt", weights_only=False)["iter"], 11)

  def test_mastery_stops_early_and_reuses_full_strength_results(self):
    state = initial_state(8)
    state.update(evaluation_streak=1, last_evaluation_iteration=10)
    with tempfile.TemporaryDirectory() as tmp:
      folder, calls = self.run_case(Path(tmp), state, iterations=10)
      status = json.loads((folder / "status.json").read_text())
      self.assertEqual(status["status"], "mastered")
      self.assertEqual(status["completed_additional_iterations"], 1)
      self.assertEqual(calls, 1)

  def test_failed_evaluation_keeps_training_checkpoint_without_promoting(self):
    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaises(RuntimeError):
        self.run_case(Path(tmp), initial_state(3), evaluation=RuntimeError("evaluation failed"))
      folder = Path(tmp) / "run"
      status = json.loads((folder / "status.json").read_text())
      self.assertEqual(status["status"], "failed")
      self.assertTrue(Path(status["latest_checkpoint"]).is_file())
      saved = torch.load(folder / "resume.pt", weights_only=False)
      self.assertEqual(saved["infos"]["koryo_curriculum"]["evaluation_streak"], 0)

  def test_taeguek_starts_from_scratch_then_resumes_its_own_state(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / "run"
      argv = [str(SCRIPT), "--motion", "taeguek01", "--output-dir", str(output),
              "--additional-iterations", "2", "--eval-every", "1", "--num-envs", "16"]
      with patch("sys.argv", argv), patch.object(driver, "run_logged", side_effect=fake_training) as train, \
           patch.object(driver, "evaluate", side_effect=fake_evaluation) as evaluate:
        driver.main()
      self.assertNotIn("--checkpoint", train.call_args_list[0].args[0])
      self.assertIn("--checkpoint", train.call_args_list[1].args[0])
      self.assertTrue(train.call_args_list[0].kwargs["tee"])
      self.assertTrue(all(c.kwargs["motion"] == "taeguek01" for c in evaluate.call_args_list))
      saved = torch.load(output / "resume.pt", weights_only=False)
      self.assertEqual(saved["iter"], 1)
      self.assertEqual(saved["infos"]["motion_identity"]["name"], "taeguek01")
      self.assertNotIn("koryo_curriculum", saved["infos"])
      self.assertEqual(saved["infos"]["taeguek01_curriculum"]["difficulty_stage"], 1)

  def test_console_tee_also_keeps_log_file(self):
    import io
    import sys
    with tempfile.TemporaryDirectory() as tmp, patch("sys.stdout", new_callable=io.StringIO) as out:
      path = Path(tmp) / "train.log"
      driver.run_logged([sys.executable, "-u", "-c", "print('Iteration time: 1.0s')"], path, tee=True)
      self.assertEqual(path.read_text(), out.getvalue())
      self.assertIn("Iteration time", out.getvalue())


if __name__ == "__main__":
  unittest.main()
