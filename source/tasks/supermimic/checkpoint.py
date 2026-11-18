"""Training output hooks for local checkpoints and deployment files."""

import json
from datetime import datetime

from supermimic.runners.supermimic_runner import SuperMimicRunner

from . import LOG_ROOT
from .checkpoint_archive import save_checkpoint
from .resume import optimizer_groups
from .training_loop import run_training_loop


class LocalBundleRunner(SuperMimicRunner):
    """Official learner with local, declarative inference metadata."""

    def __init__(self, *, manifest, **kwargs):
        self.manifest = manifest
        self._deployment_files = []
        self._checkpoint_step = -1
        super().__init__(**kwargs)

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
