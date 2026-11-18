"""Training output hooks for local checkpoints and deployment files."""

import json
from datetime import datetime

from supermimic.runners.supermimic_runner import SuperMimicRunner

from . import LOG_ROOT
from .checkpoint_archive import save_checkpoint
from .resume import optimizer_groups
from .sampler_checkpoint import sampler_snapshot
from .training_loop import run_training_loop


class LocalBundleRunner(SuperMimicRunner):
    """Official learner with local, declarative inference metadata."""

    def __init__(self, *, manifest, **kwargs):
        self.manifest = manifest
        self._deployment_files = []
        self._checkpoint_step = -1
        super().__init__(**kwargs)

    def _init_policy(self):
        if self.cfg.slot_mix.get("motion", 0) > 0:
            return super()._init_policy()

        # The native world-model environment always needs a motion reference
        # for rewards, even if every actor slot uses the velocity encoder.
        from unittest.mock import patch
        from supermimic.runners import supermimic_runner as native
        from supermimic.tasks.task_spec import make_motion_task

        motion = make_motion_task(self._representation, self._num_goals)
        build_env = native.MultiTaskWorldModelEnv

        def with_reference(*, task_specs, **kwargs):
            if kwargs["env"] is self.env and task_specs[0].name != "motion":
                task_specs = [motion, *task_specs]
            return build_env(task_specs=task_specs, **kwargs)

        with patch.object(native, "MultiTaskWorldModelEnv", with_reference):
            super()._init_policy()

    def _build_sim_obs_td(self, rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx):
        obs = super()._build_sim_obs_td(
            rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx
        )
        if self.cfg.latent_bc_weight and "goal_motion" not in obs:
            # The auxiliary loss reads the reference even for velocity-only slots.
            _, obs["goal_motion"] = self._representation.compute_wm_goal(
                self.motion, motion_times, aux_state
            )
        return obs

    def run(self):
        from .tensorboard import SuperMimicTensorBoardLogger

        try:
            return run_training_loop(self, self._checkpoint_step + 1)
        finally:
            if isinstance(self.logger, SuperMimicTensorBoardLogger):
                self.logger.close()

    def step(self):
        collected = super().step()
        self._checkpoint_step += 1
        return collected

    def checkpoint_extra_files(self, log_dir):
        # Upstream calls this after saving actor/world model, before saving
        # critic.
        # Read live models and optimizers from the same completed step.
        critic = getattr(self.policy_learner, "critic", None)
        optimizers = optimizer_groups(self)

        save_checkpoint(
            log_dir,
            step=self._checkpoint_step,
            manifest=self.manifest,
            actor=self.policy_learner.actor.state_dict(),
            world_model=self._wm.bundle(),
            critic=critic.state_dict() if critic is not None else None,
            sampler=sampler_snapshot(self),
            optimizers={
                name: [optimizer.state_dict() for optimizer in group]
                for name, group in optimizers.items()
            },
        )
        return [str(log_dir / "manifest.json"), *self._deployment_files]

    def _create_log_dir(self, *parts):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory = LOG_ROOT.joinpath(self.task_id, *parts, stamp)
        directory.mkdir(parents=True, exist_ok=False)
        print(f"[INFO] Logging to {directory}")
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(self.manifest, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(directory / "manifest.json")
        self._export_deployment_configs(directory)
        from .tensorboard import SuperMimicTensorBoardLogger

        self.logger = SuperMimicTensorBoardLogger(
            self.logger, directory, self.cfg.log_interval
        )
        return directory

    def _export_deployment_configs(self, directory):
        from .export_sim2real_cfg import (
            export_sim2real_cfg,
            validate_policy_config,
        )

        cfg = self.env.unwrapped.cfg
        validate_policy_config(cfg, self.cfg, self.manifest["task_id"])
        names = self.manifest["robot"]["joint_order"]
        self._deployment_files = []
        for spec in self._active_task_specs:
            path = export_sim2real_cfg(
                directory,
                cfg,
                self.cfg,
                self._representation,
                names,
                encoder=spec.name,
            )
            self._deployment_files.append(str(path))
            print(
                f"[INFO] Exported {spec.name} deployment configuration: {path}"
            )
