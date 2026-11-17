# Copyright 2026 The SuperMimic Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Cyclo: local validated bundles and bounded headless playback.
"""Local K1 bundle playback with the official actor and optional W&B playback."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.task_spec import build_actor_obs_td

from .checkpoint import read_bundle


@dataclass
class PlayConfig:
    motion_file: str
    checkpoint_dir: Path | None = None
    wandb_path: str | None = None
    device: str | None = None
    viewer: Literal["headless", "native", "viser"] = "viser"
    num_steps_rollout: int = 1000
    initial_task: Literal["motion", "velocity"] = "motion"
    export_policy: bool = True


def load_models(directory, rcfg, env, device):
    wm = build_world_model(rcfg, env, device)
    wm.load_bundle(
        torch.load(
            Path(directory) / "world_model.pt", map_location=device, weights_only=True
        )
    )
    actor = build_actor(rcfg, wm, device)
    actor.load_state_dict(
        torch.load(Path(directory) / "actor.pt", map_location=device, weights_only=True)
    )
    actor.eval()
    wm.dynamics.eval()
    return wm, actor


def headless_play(env, wm, actor, steps, task="motion"):
    """Actual native simulator playback, including actor-history reset boundaries."""
    actor.set_slot_mix({task: 1.0})
    obs, _ = env.reset()
    history = None
    with torch.no_grad():
        for _ in range(steps):
            state = obs["full_state"]
            rep = wm.representation.represent(state)
            aux = wm.representation.extract_aux_state(state)
            sensor = wm.representation.compute_actor_sensor(
                rep,
                wm.sim_context.read(env.unwrapped),
                aux_state=aux,
                sim_actor_obs=obs["actor"]
                if wm.representation.actor_obs_mode == "proprio"
                else None,
            )
            history = (
                sensor.unsqueeze(1).repeat(1, actor.history_length, 1)
                if history is None
                else torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)
            )
            motion = env.unwrapped.command_manager.get_term("motion")
            goals = {
                spec.name: spec.extract_training_goal(
                    motion.motion, motion.time_steps, aux
                )
                for spec in actor.task_specs
            }
            action = actor(build_actor_obs_td(goals, history))
            if not torch.isfinite(action).all():
                raise ValueError("Checkpoint produced nonfinite actions")
            obs, _, dones, _ = env.step(action)
            if dones.any():
                # Playback is single-environment; reset padding matches collection.
                history = None
    print(f"PASS local playback: {steps} native simulator steps, finite actions")


def latest_checkpoint_directory(log_root, task):
    """Find the latest saved run, skipping runs that have no complete bundle yet."""
    for directory in sorted((log_root / task).glob("*"), reverse=True):
        required_files = ("manifest.json", "actor.pt", "world_model.pt")
        if not all((directory / name).is_file() for name in required_files):
            continue
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["task_id"] == task:
                return directory
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError(
        f"No saved checkpoint bundle for {task} in {log_root / task}; "
        "use --checkpoint-dir for a run outside this directory"
    )


def play(task, cfg):
    if cfg.checkpoint_dir is not None and cfg.wandb_path is not None:
        raise ValueError("Choose only one --checkpoint-dir or --wandb-path")
    if cfg.checkpoint_dir is None and cfg.wandb_path is None:
        directory = latest_checkpoint_directory(Path("logs"), task)
        cfg = replace(cfg, checkpoint_dir=directory)
        print(f"[INFO] Selected checkpoint directory: {directory}")
    if cfg.num_steps_rollout < 1:
        raise ValueError("--num-steps-rollout must be positive")
    if cfg.wandb_path is not None:
        import wandb
        from supermimic.utils.wandb_artifacts import download_model_dir

        directory = download_model_dir(wandb.Api().run(cfg.wandb_path))
        play(task, replace(cfg, checkpoint_dir=directory, wandb_path=None))
        return
    env_cfg, rcfg = read_bundle(cfg.checkpoint_dir, task)
    env_cfg.commands["motion"].motion_file = cfg.motion_file
    if cfg.export_policy:
        from .export import export_play_policies

        export_play_policies(cfg.checkpoint_dir, task)
    device = setup_device(cfg.device)
    env = create_env(env_cfg, 1, device)
    try:
        wm, actor = load_models(cfg.checkpoint_dir, rcfg, env, device)
        if cfg.viewer == "headless":
            headless_play(env, wm, actor, cfg.num_steps_rollout, cfg.initial_task)
        else:
            from supermimic.scripts.viewers import (
                SuperMimicNativeViewer,
                SuperMimicViserViewer,
                create_wm_vis_model,
            )

            viewer = (
                SuperMimicNativeViewer
                if cfg.viewer == "native"
                else SuperMimicViserViewer
            )
            import glob

            options = (
                {
                    "motion_names": [
                        Path(p).stem for p in sorted(glob.glob(cfg.motion_file))
                    ]
                }
                if cfg.viewer == "viser"
                else {}
            )
            playback = viewer(
                env=env,
                actor=actor,
                dynamics=wm.dynamics,
                representation=wm.representation,
                wm_vis_model=create_wm_vis_model(env.unwrapped.sim.mj_model),
                num_steps_rollout=cfg.num_steps_rollout,
                sim_context=wm.sim_context,
                task_specs=actor.task_specs,
                initial_task=cfg.initial_task,
                **options,
            )
            from .viewer import adapt_ghost_visualizer

            adapt_ghost_visualizer(playback)
            playback.run()
    finally:
        env.close()


def main(argv=None):
    from .config import TASKS

    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(TASKS),
        args=argv,
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    play(task, tyro.cli(PlayConfig, args=remaining, config=mjlab.TYRO_FLAGS))
