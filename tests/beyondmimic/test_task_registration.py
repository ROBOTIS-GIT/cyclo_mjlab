"""Run with the common Python 3.12 / mjlab 1.6.1 runtime."""

import sys
import unittest

from dataclasses import asdict
from functools import partial
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("revision", ["Rev1", "Rev2"])
@pytest.mark.parametrize("motion", ["Dance1", "Dance2"])
def test_train_and_play_use_the_same_task_directory(tmp_path, monkeypatch, revision, motion):
    from types import SimpleNamespace

    from mjlab.utils.os import get_checkpoint_path
    from scripts.reinforcement_learning import train, play

    import source.tasks  # noqa: F401

    task = f"Cyclo-BeyondMimic-K1-{revision}-{motion}"
    experiment = f"cyclo_beyondmimic_k1_{revision.lower()}_{motion.lower()}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(train, "select_gpus", lambda _: (None, 0))
    runs = []
    monkeypatch.setattr(train, "run_train", lambda task, cfg, path: runs.append((cfg, path)))
    cfg = train.TrainConfig.from_task(task)
    assert cfg.agent.experiment_name == experiment
    train.launch_training(task, cfg)
    cfg, directory = runs[0]
    assert cfg.agent.experiment_name == experiment
    assert directory.parent == Path("logs/rsl_rl") / experiment
    assert task not in directory.name
    directory.mkdir(parents=True)
    checkpoint = directory / "model_20.pt"
    checkpoint.touch()
    # A later run for another motion must not affect this task's selection.
    other = directory.parent.with_name(experiment + "_other") / "2099-01-01_00-00-00"
    other.mkdir(parents=True)
    (other / "model_100.pt").touch()
    assert get_checkpoint_path(directory.parent, cfg.agent.load_run, cfg.agent.load_checkpoint) == checkpoint

    class SelectedCheckpoint(Exception):
        pass

    def select(root, **kwargs):
        assert root == directory.parent.resolve()
        assert get_checkpoint_path(root, **kwargs) == checkpoint.resolve()
        raise SelectedCheckpoint

    monkeypatch.setattr(play, "configure_torch_backends", lambda: None)
    monkeypatch.setattr(play, "load_env_cfg", lambda *_a, **_kw: SimpleNamespace(commands={}))
    monkeypatch.setattr(play, "get_checkpoint_path", select)
    with pytest.raises(SelectedCheckpoint):
        play.run_play(task, play.PlayConfig(device="cpu"))


def test_explicit_experiment_name_is_preserved(tmp_path, monkeypatch):
    from scripts.reinforcement_learning import train

    task = "Cyclo-BeyondMimic-K1-Rev2-Dance1"
    cfg = train.TrainConfig.from_task(task)
    cfg.agent.experiment_name = "custom"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(train, "select_gpus", lambda _: (None, 0))
    paths = []
    monkeypatch.setattr(train, "run_train", lambda task, cfg, path: paths.append(path))
    train.launch_training(task, cfg)
    assert paths[0].parent == Path("logs/rsl_rl/custom")


if __name__ == "__main__":
    unittest.main()
