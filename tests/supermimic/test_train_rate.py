import sys

import mjlab
import pytest
import tyro
from mjlab.tasks.registry import list_tasks
from supermimic.scripts.train import TrainConfig

from source.tasks.supermimic import train
from source.tasks.supermimic.config import TASKS, register_tasks
from source.tasks.supermimic.config.k1_rev1.base_env_cfg import make_env_cfg


def test_control_rate_allows_default_and_compensating_overrides():
    cfg = make_env_cfg()
    train.validate_control_rate(cfg)
    cfg.decimation = 8
    cfg.sim.mujoco.timestep = 0.0025
    train.validate_control_rate(cfg)


def test_cli_rejects_25hz_before_environment_construction(monkeypatch):
    if not set(TASKS).issubset(list_tasks()):
        register_tasks()

    def must_not_create_env(*args, **kwargs):
        raise AssertionError("bad reference/control rate must fail before env creation")

    monkeypatch.setattr(train, "create_env", must_not_create_env)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "Cyclo-SuperMimic-GenCoord-K1-Rev1-001",
            "--motion-file",
            "reference.npz",
            "--env.decimation",
            "8",
        ],
    )
    with pytest.raises(ValueError, match="50 Hz K1 references"):
        train.main()


def test_parsed_compensating_options_keep_50hz():
    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    cfg = tyro.cli(
        TrainConfig,
        args=["--env.decimation", "8", "--env.sim.mujoco.timestep", "0.0025"],
        default=TrainConfig.from_task("Cyclo-SuperMimic-GenCoord-K1-Rev1-001"),
        config=mjlab.TYRO_FLAGS,
    )
    train.validate_control_rate(cfg.env)


@pytest.mark.parametrize("representation", ["GenCoord", "MaxCoord"])
def test_valid_native_launcher_routes_original_runner(monkeypatch, representation):
    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    task = f"Cyclo-SuperMimic-{representation}-K1-Rev1-001"
    from types import SimpleNamespace

    sentinel = SimpleNamespace(close=lambda: None)
    calls = []

    def create_env(cfg, num_envs, device):
        train.validate_control_rate(cfg)
        assert cfg.commands["motion"].motion_file == "validated.npz"
        assert cfg.commands["motion"].sampling_mode == "external"
        assert num_envs == 2 and device == "cpu"
        return sentinel

    class Runner:
        def __init__(self, **kwargs):
            assert kwargs["env"] is sentinel
            assert kwargs["task_id"] == task
            assert kwargs["registry_name"] is None
            assert kwargs["cfg"].representation == representation.lower()

        def run(self):
            calls.append("run")

    monkeypatch.setattr(train, "create_env", create_env)
    monkeypatch.setattr(train, "setup_device", lambda device: "cpu")
    monkeypatch.setattr(train, "SuperMimicRunner", Runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            task,
            "--motion-file",
            "validated.npz",
            "--num-envs",
            "2",
            "--device",
            "cpu",
        ],
    )
    train.main()
    assert calls == ["run"]
