import json

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from source.tasks.blinkmimic.tensorboard import BlinkMimicTensorBoard, wait_with_tensorboard


def make_run(tmp_path):
    (tmp_path / "run-record.json").write_text(json.dumps({"started_at": "2026-11-17T00:00:00+00:00"}))
    return tmp_path / "train_log.csv"


def test_complete_rows_are_mirrored_once_with_common_tags(tmp_path):
    csv = make_run(tmp_path)
    metrics = BlinkMimicTensorBoard(tmp_path)
    metrics.update()  # Trainer has not created the CSV yet.
    csv.write_text("iteration,wall_h,sim_rew,ep_len,wm_loss,ep_ret_norm\n50,0.1,0.8,120,0.2,0.7\n100,0.2")
    metrics.update()
    metrics.update()
    with csv.open("a") as out:
        out.write(",0.9,130,0.1,0.85\n")
    metrics.update()
    metrics.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    rewards = events.Scalars("Train/mean_reward")
    assert [e.step for e in rewards] == [50, 100]
    assert [e.value for e in rewards] == pytest.approx([.8, .9])
    assert rewards[1].wall_time - rewards[0].wall_time == pytest.approx(360)
    assert events.Scalars("Train/mean_episode_length")[-1].value == 130
    assert events.Scalars("Loss/world_model")[-1].value == pytest.approx(.1)
    assert events.Scalars("BlinkMimic/normalized_episode_return")[-1].value == pytest.approx(.85)


def test_wait_records_final_row_after_trainer_exit(tmp_path):
    csv = make_run(tmp_path)
    csv.write_text("iteration,wall_h,sim_rew\n10,0.1,0.5\n")
    class Finished:
        returncode = 0
        def poll(self):
            return self.returncode
    assert wait_with_tensorboard(Finished(), tmp_path) == 0
    assert EventAccumulator(str(tmp_path)).Reload().Scalars("Train/mean_reward")[0].step == 10
