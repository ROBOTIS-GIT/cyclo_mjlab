"""Training defaults must not alter shared learner or saved bundle contracts."""

from dataclasses import asdict

import mjlab
import pytest
import tyro
from supermimic.scripts.train import TrainConfig

from source.tasks.supermimic.config import TASKS, register_tasks
from source.tasks.supermimic.train import training_defaults


@pytest.mark.parametrize("task", TASKS)
def test_training_defaults_only_change_observation_and_task_representation(task):
    from mjlab.tasks.registry import list_tasks

    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    original = TrainConfig.from_task(task)
    cfg = training_defaults(task)
    representation, _ = TASKS[task]
    assert cfg.runner.actor_obs == "proprio"
    assert cfg.runner.history_length == 10
    assert cfg.runner.representation == representation
    expected = asdict(original.runner)
    expected.update(actor_obs="proprio", history_length=10, representation=representation)
    assert asdict(cfg.runner) == expected
    parsed = tyro.cli(
        TrainConfig,
        args=["--runner.actor-obs", "rep_state", "--runner.history-length", "1"],
        default=cfg,
        config=mjlab.TYRO_FLAGS,
    )
    assert parsed.runner.actor_obs == "rep_state"
    assert parsed.runner.history_length == 1
    assert TrainConfig.from_task(task).runner.actor_obs == "rep_state"


def test_custom_task_uses_configured_representation(monkeypatch):
    from source.tasks.supermimic.config import check_representation_matches
    from source.tasks.supermimic.config.k1_rev1 import task_001
    from mjlab.tasks.registry import load_env_cfg

    def custom_env(representation, play=False):
        cfg = task_001.make_env_cfg(representation, play=play)
        cfg.commands["motion"].motion_file = ["custom_motion.npz"]
        return cfg

    task = "Cyclo-SuperMimic-K1-Rev1-Custom"
    monkeypatch.setitem(TASKS, task, ("maxcoord", custom_env))
    register_tasks()
    register_tasks()
    for play in (False, True):
        assert load_env_cfg(task, play=play).commands["motion"].motion_file == ["custom_motion.npz"]
    assert load_env_cfg(task, play=True).scene.num_envs == 1
    assert training_defaults(task).runner.representation == "maxcoord"
    check_representation_matches(task, "maxcoord")
    with pytest.raises(ValueError, match="maxcoord"):
        check_representation_matches(task, "gencoord")


def test_registry_does_not_borrow_robot_or_algorithm_settings():
    from mjlab.rl import RslRlBaseRunnerCfg
    from mjlab.tasks.registry import load_rl_cfg
    from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

    register_tasks()
    for task in TASKS:
        assert type(load_rl_cfg(task)) is RslRlBaseRunnerCfg
        assert isinstance(training_defaults(task).runner, SuperMimicRunnerCfg)
