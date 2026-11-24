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

"""
Restore training models and optimizer states before the first learner update.
"""


from dataclasses import asdict

from supermimic.runners.run_config import load_runner_cfg

from .archive import check_training_contract, initialize_world_model
from .manifest import check_actor_observations


def validate_resume_configuration(checkpoint, manifest):
    if checkpoint.step is None:
        raise ValueError(
            "Training resume requires last.pt or a numbered combined"
            " checkpoint"
        )
    check_training_contract(checkpoint.manifest, manifest)
    saved, current = (asdict(load_runner_cfg(value)) for value in (checkpoint.manifest, manifest))
    if saved["actor_obs"] == "proprio":
        check_actor_observations(
            checkpoint.manifest["environment"].get("observations", {}).get("actor"),
            manifest["environment"]["observations"].get("actor"),
        )
    # Logging cadence and the total iteration target may change. Architecture,
    # losses and optimizer settings remain those of the saved training run.
    for name in (
        "headless",
        "max_iterations",
        "run_name",
        "log_interval",
        "save_interval",
        "wandb_run_path",
        "wandb_project",
    ):
        saved.pop(name, None)
        current.pop(name, None)
    if saved != current:
        raise ValueError(
            "Resume learner configuration differs from the checkpoint"
        )


def optimizer_groups(runner):
    groups = {
        "actor": runner.policy_learner.actor_optimizers,
        "world_model": runner.world_model_learner.optimizers,
    }
    if getattr(runner.policy_learner, "critic", None) is not None:
        groups["critic"] = runner.policy_learner.critic_optimizers
    return groups


def restore_training_state(runner, checkpoint):
    critic = getattr(runner.policy_learner, "critic", None)
    if (critic is None) != (checkpoint.critic is None):
        raise ValueError(
            "Checkpoint critic does not match the selected learner"
        )
    groups = optimizer_groups(runner)
    if checkpoint.optimizers is not None:
        if set(groups) != set(checkpoint.optimizers):
            raise ValueError(
                "Checkpoint optimizer groups do not match the learner"
            )
        for name, optimizers in groups.items():
            if len(optimizers) != len(checkpoint.optimizers[name]):
                raise ValueError(
                    f"Checkpoint {name} optimizer count does not match"
                )
    initialize_world_model(runner._wm, checkpoint.world_model)
    runner.policy_learner.actor.load_state_dict(checkpoint.actor)
    if critic is not None:
        critic.load_state_dict(checkpoint.critic)
    if checkpoint.optimizers is None:
        print(
            "[WARN] Model-only resume: optimizers restart from their"
            " configured defaults."
        )
    else:
        for name, optimizers in groups.items():
            for optimizer, saved in zip(
                optimizers, checkpoint.optimizers[name], strict=True
            ):
                restore_optimizer(optimizer, saved)
    from .sampler import restore_sampler

    restore_sampler(runner, checkpoint.sampler)
    runner._checkpoint_step = checkpoint.step
    print(
        f"[INFO] Resumed {checkpoint.source}; next"
        f" iteration={checkpoint.step + 1}. Replay buffer and"
        " simulation state restart."
    )


def restore_optimizer(optimizer, saved):
    """Keep device LR tensors used by subsequently captured CUDA graphs."""
    import torch

    learning_rates = [group["lr"] for group in optimizer.param_groups]
    optimizer.load_state_dict(saved)
    for group, original_lr in zip(
        optimizer.param_groups, learning_rates, strict=True
    ):
        if torch.is_tensor(original_lr):
            original_lr.copy_(
                torch.as_tensor(group["lr"], device=original_lr.device)
            )
            group["lr"] = original_lr
