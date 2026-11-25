import time

import pytest
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from source.tasks.blinkmimic.runtime.tensorboard import BlinkMimicTensorBoard


def test_complete_rows_are_mirrored_once_with_common_tags(tmp_path):
    csv = tmp_path / "train_log.csv"
    before = time.time()
    metrics = BlinkMimicTensorBoard(tmp_path)
    metrics.update()  # Trainer has not created the CSV yet.
    csv.write_text(
        "iteration,wall_h,sim_rew,ep_len,wm_loss,ep_ret_norm\n"
        "50,0.1,0.8,120,0.2,0.7\n100,0.2"
    )
    metrics.update()
    metrics.update()
    with csv.open("a") as out:
        out.write(",0.9,130,0.1,0.85\n")
    metrics.update()
    metrics.close()
    after = time.time()
    events = EventAccumulator(str(tmp_path)).Reload()
    rewards = events.Scalars("Train/mean_reward")
    assert [e.step for e in rewards] == [50, 100]
    assert [e.value for e in rewards] == pytest.approx([0.8, 0.9])
    assert all(before <= event.wall_time <= after for event in rewards)
    assert [event.value for event in events.Scalars("Perf/wall_hours")] == pytest.approx([0.1, 0.2])
    assert events.Scalars("Train/mean_episode_length")[-1].value == 130
    assert events.Scalars("Loss/world_model")[-1].value == pytest.approx(0.1)
    assert events.Scalars("BlinkMimic/normalized_episode_return")[
        -1
    ].value == pytest.approx(0.85)


def test_wait_records_final_row_after_trainer_exit(tmp_path):
    from source.tasks.blinkmimic.cli import wait_for_training

    csv = tmp_path / "train_log.csv"
    csv.write_text("iteration,wall_h,sim_rew\n10,0.1,0.5\n")
    (tmp_path / "ckpt_001000.pt").write_bytes(b"published checkpoint")
    (tmp_path / "ckpt_002000.pt.tmp").write_bytes(b"partial checkpoint")
    (tmp_path / "last.pt").write_bytes(b"latest checkpoint")

    class Finished:
        returncode = 0

        def poll(self):
            return self.returncode

    assert wait_for_training(Finished(), tmp_path) == 0
    assert (tmp_path / "model_1000.pt").read_bytes() == b"published checkpoint"
    assert not (tmp_path / "ckpt_001000.pt").exists()
    assert (
        tmp_path / "ckpt_002000.pt.tmp"
    ).read_bytes() == b"partial checkpoint"
    assert (tmp_path / "last.pt").read_bytes() == b"latest checkpoint"
    assert (
        EventAccumulator(str(tmp_path))
        .Reload()
        .Scalars("Train/mean_reward")[0]
        .step
        == 10
    )


def test_detached_resume_logs_include_first_and_second_update(tmp_path):
    from source.tasks.blinkmimic.cli import initialize_resume_log

    csv = tmp_path / "train_log.csv"
    initialize_resume_log(tmp_path)
    with csv.open("a") as stream:
        for iteration in (5010, 5020):
            stream.write(f"{iteration}," + ",".join(["0"] * 18) + "\n")
    metrics = BlinkMimicTensorBoard(tmp_path)
    metrics.update()
    metrics.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [event.step for event in events.Scalars("Train/mean_reward")] == [5010, 5020]
