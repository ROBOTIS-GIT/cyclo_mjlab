"""Run with the existing Python 3.11 / mjlab 1.2 runtime."""

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


class BeyondMimicCompatibility(unittest.TestCase):
    def test_alias_configs_are_identical(self):
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

        import source.tasks  # noqa: F401

        for dance in ("Dance1", "Dance2"):
            old = f"Cyclo-Mimic-K1-Rev1-{dance}"
            explicit = f"Cyclo-BeyondMimic-K1-Rev1-{dance}"
            for play in (False, True):
                self.assertEqual(
                    config_value(asdict(load_env_cfg(old, play=play))),
                    config_value(asdict(load_env_cfg(explicit, play=play))),
                )
            self.assertEqual(
                config_value(asdict(load_rl_cfg(old))),
                config_value(asdict(load_rl_cfg(explicit))),
            )
        self.assertNotIn("source.tasks.supermimic.tasks", sys.modules)


if __name__ == "__main__":
    unittest.main()
