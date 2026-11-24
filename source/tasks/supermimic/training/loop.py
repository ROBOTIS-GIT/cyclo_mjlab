# Copyright 2026 The SuperMimic Authors.
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
# Authors: The SuperMimic Authors, Kiwoong Park

"""Native training loop with an explicit start iteration for local resumes.

Adapted from supermimic.runners.supermimic_runner.SuperMimicRunner.run.
Collection and learner updates remain upstream methods; no upstream edits.
"""

import os
import time
from copy import copy
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
import wandb
from supermimic.envs.world_model_env import sample_start
from supermimic.runners.supermimic_runner import (
    Ratio,
    export_actor_as_onnx,
)

from source.utils.wandb_models import model_upload_requested


def log_model_artifact(files, run_dir):
    """Publish the run-relative paths with SuperMimic's model version retention."""
    run = wandb.run
    name = f"model-{run.id}"
    try:
        artifact = wandb.Artifact(name, type="model")
        root = Path(run_dir).resolve()
        for file in files:
            artifact.add_file(file, name=Path(file).resolve().relative_to(root).as_posix())
        run.log_artifact(artifact)
        collection = f"{run.entity}/{run.project}/{name}"
        versions = sorted(wandb.Api().artifacts("model", collection), key=lambda a: int(a.version[1:]))
        for stale in versions[:-5]:
            if int(stale.version[1:]) % 25 and not stale.aliases:
                stale.delete()
    except Exception as error:
        print(f"[WARN] Checkpoint artifact log/prune failed: {error}")


def export_training_policies(runner, log_dir, run_name):
    if runner._representation.actor_obs_mode != "proprio" or runner.motion.time_step_total > 15_000:
        return []
    # The native exporter enumerates task_specs, including for velocity replay.
    # Share the weights and limit the export view to trained slots.
    actor = copy(runner.policy_learner.actor)
    actor.task_specs = runner._active_task_specs
    return export_actor_as_onnx(
        actor=actor,
        env=runner.env.unwrapped,
        path=str(log_dir),
        run_name=run_name,
        term_widths=runner._representation.proprio_term_widths(runner._num_actions),
    )


def run_training_loop(runner, start_iteration=0) -> None:
    log_dir = runner._create_log_dir()
    os.environ.setdefault("WANDB_MODE", "online" if model_upload_requested() else "disabled")
    runner._init_wandb(log_dir)

    runner.env.reset()
    runner._seed_sim_history()

    print(
        f"[INFO] Training iterations [{start_iteration},"
        f" {runner.cfg.max_iterations})"
    )

    should_train = Ratio(
        runner.env.num_envs
        / runner.cfg.wm.num_batches
        / runner.cfg.wm.batch_size
        / runner.cfg.wm.horizon
        * runner.cfg.replay_ratio
    )

    num_collected_steps = 0
    start_time = time.time()
    wm_learn_time = 0.0
    policy_learn_time = 0.0
    loss_dict: dict[str, torch.Tensor] = {}
    upload_models = (
        model_upload_requested() and wandb.run.settings.mode == "online"
    )
    uploader = ThreadPoolExecutor(max_workers=1) if upload_models else None
    upload: Future | None = None

    try:
        for step in range(start_iteration, runner.cfg.max_iterations):
            num_collected_steps += runner.step()

            for _ in range(should_train(step)):
                if (
                    runner.replay_buffer.num_transitions
                    < runner.cfg.wm.horizon
                ):
                    continue

                data = (
                    runner.replay_buffer.sample(
                        runner.cfg.wm.horizon, runner.cfg.wm.batch_size
                    )
                    for _ in range(runner.cfg.wm.num_batches)
                )

                start_wm_time = time.time()
                wm_logs = runner.world_model_learner.update(data)
                wm_learn_time += time.time() - start_wm_time

                start_policy_time = time.time()
                policy_cfg = runner.cfg.policy
                starts = (
                    sample_start(
                        runner.replay_buffer,
                        runner.cfg.history_length,
                        policy_cfg.batch_size,
                    )
                    for _ in range(policy_cfg.num_batches)
                )
                policy_logs = runner.policy_learner.update(starts)
                policy_learn_time += time.time() - start_policy_time

                loss_dict.update(wm_logs)
                loss_dict.update(policy_logs)

            if step % runner.cfg.log_interval == 0:
                elapsed_time = time.time() - start_time
                total_train_time = wm_learn_time + policy_learn_time
                collect_time = elapsed_time - total_train_time

                runner.logger.log(
                    it=step // runner.cfg.log_interval,
                    total_it=runner.cfg.max_iterations
                    // runner.cfg.log_interval,
                    collect_time=collect_time,
                    wm_learn_time=wm_learn_time,
                    policy_learn_time=policy_learn_time,
                    loss_dict=loss_dict,
                    num_collected_steps=num_collected_steps,
                )

                start_time = time.time()
                wm_learn_time = 0
                policy_learn_time = 0
                num_collected_steps = 0

            if (
                step % runner.cfg.save_interval == 0
                or step == runner.cfg.max_iterations - 1
            ):
                assert wandb.run is not None  # _init_wandb ran before the loop
                # The previous version must be logged before its files are
                # overwritten.
                if upload is not None:
                    upload.result()
                world_model_path = str(log_dir / "world_model.pt")
                actor_path = str(log_dir / "actor.pt")
                torch.save(runner._wm.bundle(), world_model_path)
                torch.save(
                    runner.policy_learner.actor.state_dict(), actor_path
                )
                checkpoint_files = [world_model_path, actor_path]
                checkpoint_files += runner.checkpoint_extra_files(log_dir)
                critic = getattr(runner.policy_learner, "critic", None)
                if critic is not None:
                    critic_path = str(log_dir / "critic.pt")
                    torch.save(critic.state_dict(), critic_path)
                    checkpoint_files.append(critic_path)
                checkpoint_files += export_training_policies(runner, log_dir, wandb.run.name)
                if uploader is not None:
                    upload = uploader.submit(
                        log_model_artifact, checkpoint_files, log_dir
                    )

        if upload is not None:
            upload.result()
    finally:
        if uploader is not None:
            uploader.shutdown()
        runner._cleanup()
