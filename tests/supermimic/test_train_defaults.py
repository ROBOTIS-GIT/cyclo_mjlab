"""Training defaults must not alter shared learner or saved bundle contracts."""

from dataclasses import asdict

import mjlab
import pytest
import tyro
from supermimic.scripts.train import TrainConfig

from source.tasks.supermimic.tasks import TASKS, register_tasks
from source.tasks.supermimic.train import training_defaults


@pytest.mark.parametrize("task", TASKS)
def test_training_defaults_only_change_observation_and_task_representation(task):
    from mjlab.tasks.registry import list_tasks

    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    original = TrainConfig.from_task(task)
    cfg = training_defaults(task)
    assert cfg.runner.actor_obs == "proprio"
    assert cfg.runner.history_length == 10
    assert cfg.runner.representation == TASKS[task]
    expected = asdict(original.runner)
    expected.update(actor_obs="proprio", history_length=10, representation=TASKS[task])
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
