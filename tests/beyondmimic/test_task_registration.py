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


if __name__ == "__main__":
    unittest.main()
