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

"""TensorBoard output alongside the official SuperMimic console/W&B logger."""

import statistics

import torch
from torch.utils.tensorboard import SummaryWriter


class SuperMimicTensorBoardLogger:
    def __init__(self, logger, run_dir, log_interval):
        self.logger = logger
        self.log_interval = log_interval
        self.writer = SummaryWriter(log_dir=str(run_dir), flush_secs=10)

    def process_env_step(self, rewards, terminated, extras):
        self.logger.process_env_step(rewards, terminated, extras)

    def log(self, it, total_it, collect_time, wm_learn_time, policy_learn_time,
            loss_dict, num_collected_steps, width=80, pad=40):
        # Resolve GPU scalars once; the original logger then receives host values.
        keys = [key for key, value in loss_dict.items() if isinstance(value, torch.Tensor)]
        values = torch.stack([loss_dict[key] for key in keys]).tolist() if keys else []
        losses = loss_dict | dict(zip(keys, values, strict=True))
        extras = self.logger._episode_extras()
        self.logger.log(it, total_it, collect_time, wm_learn_time, policy_learn_time,
                        losses, num_collected_steps, width, pad)
        step = it * self.log_interval
        for name, value in losses.items():
            self.writer.add_scalar(name if name.startswith("Loss/") else "Loss/" + name, value, step)
        for name, value in extras.items():
            self.writer.add_scalar(name if "/" in name else "Episode/" + name, value, step)
        duration = collect_time + wm_learn_time + policy_learn_time
        metrics = {
            "Perf/total_fps": int(num_collected_steps / (duration + 1e-6)),
            "Perf/collection_time": collect_time,
            "Perf/learning_time": wm_learn_time + policy_learn_time,
            "Perf/world_model_learning_time": wm_learn_time,
            "Perf/policy_learning_time": policy_learn_time,
            "Perf/total_steps": self.logger.tot_timesteps,
        }
        if self.logger.rewbuffer:
            metrics["Train/mean_reward"] = statistics.mean(self.logger.rewbuffer)
            metrics["Train/mean_episode_length"] = statistics.mean(self.logger.lenbuffer)
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def close(self):
        self.writer.close()
