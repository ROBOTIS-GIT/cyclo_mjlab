# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Kiwoong Park

"""Incrementally mirror upstream CSV metrics without changing the learner."""

import csv
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

SCALAR_TAGS = {
    "sim_rew": "Train/mean_reward",
    "ep_len": "Train/mean_episode_length",
    "ep_ret_norm": "BlinkMimic/normalized_episode_return",
    "term_frac": "BlinkMimic/termination_fraction",
    "wm_loss": "Loss/world_model",
    "wm_err8": "BlinkMimic/world_model_error_8_steps",
    "pi_loss": "Loss/policy",
    "v_loss": "Loss/value_function",
    "lr": "Loss/learning_rate",
    "pi_lr": "Loss/policy_learning_rate",
    "pi_gradnorm": "Policy/gradient_norm",
    "pi_mult": "Policy/learning_rate_multiplier",
    "imag_rew": "BlinkMimic/imagined_reward",
    "imag_survival": "BlinkMimic/imagined_survival",
    "nonfinite": "BlinkMimic/nonfinite_updates",
    "sim_steps": "Perf/total_steps",
    "it_per_s": "Perf/iterations_per_second",
    "wall_h": "Perf/wall_hours",
}


class BlinkMimicTensorBoard:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.writer = SummaryWriter(log_dir=str(self.run_dir), flush_secs=10)
        self.writer.add_text(
            "Metrics/reward_definition",
            "Train/mean_reward is BlinkMimic's normalized mean reward per"
            " simulation step (sim_rew), not BeyondMimic's mean episode"
            " return. Episode return cannot be reconstructed exactly from the"
            " aggregated CSV.",
        )
        self.offset = 0
        self.fields = None
        self.last_iteration = -1

    def update(self):
        path = self.run_dir / "train_log.csv"
        if not path.exists():
            return
        with path.open() as source:
            source.seek(self.offset)
            while True:
                line = source.readline()
                if not line.endswith("\n"):
                    break  # Do not consume a partially written CSV row.
                self.offset = source.tell()
                values = next(csv.reader([line]))
                if self.fields is None:
                    self.fields = values
                    continue
                row = dict(zip(self.fields, values, strict=True))
                iteration = int(row["iteration"])
                if iteration <= self.last_iteration:
                    continue
                for column, tag in SCALAR_TAGS.items():
                    if column in row and math.isfinite(
                        value := float(row[column])
                    ):
                        self.writer.add_scalar(tag, value, iteration)
                self.last_iteration = iteration
        self.writer.flush()

    def close(self):
        self.writer.close()
