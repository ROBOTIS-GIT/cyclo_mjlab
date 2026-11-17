import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from supermimic.utils.logger import Logger
from source.tasks.supermimic.tensorboard import SuperMimicTensorBoardLogger


def test_tensorboard_preserves_upstream_episode_statistics_and_wandb(tmp_path, monkeypatch):
    rows = []
    monkeypatch.setattr("supermimic.utils.logger.wandb.log", lambda row, step: rows.append((row, step)))
    base = Logger("test", 2, 100, "cpu")
    logger = SuperMimicTensorBoardLogger(base, tmp_path, 10)
    logger.process_env_step(torch.tensor([1., 2.]), torch.tensor([False, False]), {})
    logger.process_env_step(torch.tensor([3., 4.]), torch.tensor([True, True]),
                            {"log": {"tracking": torch.tensor([.2, .4])}})
    logger.log(3, 100, .1, .2, .3, {"world_model": torch.tensor(.5)}, 4)
    logger.close()
    assert rows[0][1] == 3
    assert rows[0][0]["Train/mean_reward"] == 5  # Mean of episode sums 4 and 6.
    assert rows[0][0]["world_model"] == pytest.approx(.5)
    events = EventAccumulator(str(tmp_path)).Reload()
    reward = events.Scalars("Train/mean_reward")[0]
    assert reward.step == 30 and reward.value == 5
    assert events.Scalars("Train/mean_episode_length")[0].value == 2
    assert events.Scalars("Loss/world_model")[0].value == pytest.approx(.5)
    assert events.Scalars("Episode/tracking")[0].value == pytest.approx(.3)
    assert events.Scalars("Perf/learning_time")[0].value == pytest.approx(.5)
    assert base.ep_extras == []
