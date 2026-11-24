"""Run with the common Python 3.12 / mjlab 1.6.1 runtime."""

import sys
import unittest
from dataclasses import asdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def config_value(value):
    if isinstance(value, partial):
        return (
            config_value(value.func),
            config_value(value.args),
            config_value(value.keywords),
        )
    if callable(value):
        return (value.__module__, value.__qualname__)
    if isinstance(value, dict):
        return {k: config_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [config_value(v) for v in value]
    return value


class BeyondMimicRegistration(unittest.TestCase):
    def test_registered_configs_preserve_training_and_play_settings(self):
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

        import source.tasks  # noqa: F401

        from source.tasks.beyondmimic.config.k1_rev1.dance1_env_cfg import k1_rev1_dance1_env_cfg
        from source.tasks.beyondmimic.config.k1_rev1.dance2_env_cfg import k1_rev1_dance2_env_cfg
        from source.tasks.beyondmimic.config.k1_rev1.agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg

        for dance, make_cfg in (
            ("Dance1", k1_rev1_dance1_env_cfg), ("Dance2", k1_rev1_dance2_env_cfg),
        ):
            explicit = f"Cyclo-BeyondMimic-K1-Rev1-{dance}"
            for play in (False, True):
                self.assertEqual(
                    config_value(asdict(make_cfg(play=play))),
                    config_value(asdict(load_env_cfg(explicit, play=play))),
                )
            self.assertEqual(
                config_value(asdict(k1_rev1_mimic_ppo_runner_cfg())),
                config_value(asdict(load_rl_cfg(explicit))),
            )


def test_tracking_checkpoint_selection_in_shared_experiment(tmp_path):
    import pytest
    from source import SRC_PATH
    from source.tasks.beyondmimic.checkpoint import tracking_checkpoint_path

    task = "Cyclo-BeyondMimic-K1-Rev2-Dance1"
    other = task.replace("Dance1", "Dance2")
    motion = SRC_PATH / "assets/motions/K1_rev2/dance1/dance1.npz"

    def run(name, legacy_motion=None):
        directory = tmp_path / name
        directory.mkdir()
        if legacy_motion:
            (directory / "params").mkdir()
            (directory / "params/env.yaml").write_text(
                f"commands:\n  reference_trajectory:\n    trajectory_file: {legacy_motion}\n"
            )
        path = directory / "model_20.pt"
        path.touch()
        return path

    old = run("2026-11-21_10-00-00", "/workspace/cyclo_mjlab/source/assets/motions/K1_rev2/dance1/dance1.npz")
    run("2026-11-22_10-00-00", str(motion).replace("dance1", "dance2"))
    run("2026-11-23_10-00-00_custom")

    def select(pattern=".*"):
        return tracking_checkpoint_path(tmp_path, task, motion, pattern, r"model_\d+\.pt")

    assert select() == old
    new = run("2026-11-24_10-00-00_" + task + "_custom")
    dance2 = run("2026-11-24_11-00-00_" + other)
    (tmp_path / ("2026-11-25_10-00-00_" + task)).mkdir()
    assert select() == new
    assert select(dance2.parent.name) == dance2
    with pytest.raises(ValueError, match="No matching checkpoint"):
        select("2026-11-23.*")


if __name__ == "__main__":
    unittest.main()
