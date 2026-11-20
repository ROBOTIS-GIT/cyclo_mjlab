# Copyright 2026 The SuperMimic Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Cyclo: local validated bundles and bounded headless playback.
"""
Local K1 bundle playback with the official actor and optional W&B playback.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.task_spec import build_actor_obs_td

from . import LOG_ROOT
from .config import TASKS, check_representation_matches
from .checkpoint_archive import load_checkpoint


@dataclass
class PlayConfig:
    motion_file: str
    checkpoint_dir: Path | None = None
    checkpoint: Path | None = None
    wandb_path: str | None = None
    device: str | None = None
    viewer: Literal["headless", "native", "viser"] = "viser"
    num_steps_rollout: int = 1000
    initial_task: Literal["motion", "velocity"] = "motion"
    export_policy: bool = True


def load_models(directory, rcfg, env, device):
    checkpoint = load_checkpoint(directory)
    wm = build_world_model(rcfg, env, device)
    wm.load_bundle(checkpoint.world_model)
    actor = build_actor(rcfg, wm, device)
    actor.load_state_dict(checkpoint.actor)
    actor.eval()
    wm.dynamics.eval()
    return wm, actor


def headless_play(env, wm, actor, steps, task="motion"):
    """
    Actual native simulator playback, including actor-history reset boundaries.
    """
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
                sim_actor_obs=(
                    obs["actor"]
                    if wm.representation.actor_obs_mode == "proprio"
                    else None
                ),
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
                # Playback is single-environment; reset padding matches
                # collection.
                history = None
    print(
        f"PASS local playback: {steps} native simulator steps, finite actions"
    )


def latest_checkpoint_directory(log_root, task):
    """
    Find the latest saved run, skipping runs that have no complete bundle yet.
    """
    for directory in sorted((log_root / task).glob("*"), reverse=True):
        required_files = (
            ("manifest.json", "last.pt")
            if (directory / "last.pt").is_file()
            else ("manifest.json", "actor.pt", "world_model.pt")
        )
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


def resolve_playback_checkpoint(task, cfg):
    if (
        sum(
            value is not None
            for value in (cfg.checkpoint, cfg.checkpoint_dir, cfg.wandb_path)
        )
        > 1
    ):
        raise ValueError(
            "Choose only one --checkpoint, --checkpoint-dir or --wandb-path"
        )
    if (
        cfg.checkpoint is None
        and cfg.checkpoint_dir is None
        and cfg.wandb_path is None
    ):
        directory = latest_checkpoint_directory(LOG_ROOT, task)
        cfg = replace(cfg, checkpoint_dir=directory)
        print(f"[INFO] Selected checkpoint directory: {directory}")
    if cfg.num_steps_rollout < 1:
        raise ValueError("--num-steps-rollout must be positive")
    if cfg.wandb_path is not None:
        import wandb
        from supermimic.utils.wandb_artifacts import download_model_dir

        directory = download_model_dir(wandb.Api().run(cfg.wandb_path))
        return load_checkpoint(directory)
    return load_checkpoint(
        cfg.checkpoint if cfg.checkpoint is not None else cfg.checkpoint_dir,
    )


def run_interactive_playback(env, wm, actor, cfg):
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


def play(task, cfg):
    checkpoint = resolve_playback_checkpoint(task, cfg)
    rcfg = load_runner_cfg(checkpoint.manifest)
    check_representation_matches(task, rcfg.representation)
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.commands["motion"].motion_file = cfg.motion_file
    if cfg.export_policy:
        from .export import export_play_policies

        source_task = checkpoint.manifest["task_id"]
        if source_task in TASKS:
            export_play_policies(checkpoint, source_task)
        else:
            print(f"[INFO] Skipping automatic ONNX export: training task {source_task!r} is unregistered.")
    device = setup_device(cfg.device)
    env = create_env(env_cfg, 1, device)
    try:
        wm, actor = load_models(checkpoint, rcfg, env, device)
        if cfg.viewer == "headless":
            headless_play(
                env, wm, actor, cfg.num_steps_rollout, cfg.initial_task
            )
        else:
            run_interactive_playback(env, wm, actor, cfg)
    finally:
        env.close()


def main(argv=None):
    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(TASKS),
        args=argv,
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    play(task, tyro.cli(PlayConfig, args=remaining, config=mjlab.TYRO_FLAGS))
