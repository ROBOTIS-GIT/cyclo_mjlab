# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a taekwondo checkpoint without clip-end teleport credit."""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import MethodType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import numpy as np
import source.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from source.tasks.mimic.mdp.koryo_curriculum import KoryoObservationNoise, scaled_ranges
from source.tasks.mimic.motion_curriculum import MOTIONS, motion_identity, validate_checkpoint_motion


def configure_disturbances(cfg, noise_scale, push_scale):
  """Match the curriculum amplitudes without its training start sampler/gate."""
  cmd = cfg.commands["reference_trajectory"]
  cmd.reset_pose_noise = scaled_ranges(cmd.reset_pose_noise, noise_scale)
  cmd.reset_velocity_noise = scaled_ranges(cmd.reset_velocity_noise, noise_scale)
  cmd.reset_joint_position_noise = tuple(x * noise_scale for x in cmd.reset_joint_position_noise)
  for term in cfg.observations["actor"].terms.values():
    if term.noise is not None:
      term.noise = KoryoObservationNoise(
        n_min=term.noise.n_min, n_max=term.noise.n_max,
        operation=term.noise.operation, curriculum_scale=noise_scale,
      )
  material = cfg.events["physics_material"].params
  midpoint = sum(material["ranges"]) / 2
  material["ranges"] = tuple(midpoint + (x-midpoint)*noise_scale for x in material["ranges"])
  com = cfg.events["torso_com_offset_noise"].params
  com["com_range"] = scaled_ranges(com["com_range"], noise_scale)
  encoder = cfg.events["joint_home_offset_noise"].params
  encoder["pos_distribution_params"] = tuple(x*noise_scale for x in encoder["pos_distribution_params"])
  push = cfg.events["push_robot"].params
  push["velocity_range"] = scaled_ranges(push["velocity_range"], push_scale)


def evaluation_window(identity, full_motion, trials, start_times, duration):
  """Derive the evaluation horizon and valid starts from the selected NPZ."""
  if full_motion:
    return [0.0] * trials, identity["frames"] / identity["fps"] + 0.1
  starts = list(range(int(np.ceil(identity["duration_s"])))) if start_times is None else start_times
  if not 0 < duration < float("inf") or any(not 0 <= t < identity["duration_s"] for t in starts):
    raise ValueError(f"Duration must be positive and finite; starts must be in [0, {identity['duration_s']}).")
  return starts, duration


def main(default_motion="koryo"):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion", choices=tuple(MOTIONS), default=default_motion)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--condition", choices=("train", "clean"), default="clean")
  parser.add_argument("--start-times", type=float, nargs="+")
  parser.add_argument("--duration", type=float, default=30.0)
  parser.add_argument("--full-motion", action="store_true", help="Start at frame zero and attempt the entire clip.")
  parser.add_argument("--trials", type=int, default=32, help="Parallel attempts with --full-motion.")
  parser.add_argument("--noise-scale", type=float, default=1.0)
  parser.add_argument("--push-scale", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error("Checkpoint does not exist.")
  if args.trials < 1 or not 0 <= args.noise_scale <= 1 or not 0 <= args.push_scale <= 1:
    parser.error("Trials must be positive; disturbance scales must be in [0, 1].")
  identity = motion_identity(args.motion)
  try:
    args.start_times, args.duration = evaluation_window(
      identity, args.full_motion, args.trials, args.start_times, args.duration)
  except ValueError as error:
    parser.error(str(error))
  configure_torch_backends()
  task = MOTIONS[args.motion].task
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = len(args.start_times)
  cfg.episode_length_s = args.duration
  cfg.seed = args.seed
  cmdcfg = cfg.commands["reference_trajectory"]
  cmdcfg.debug_vis = False
  configure_disturbances(cfg, args.noise_scale, args.push_scale)
  if args.condition == "clean":
    cfg.events = {}
    cfg.observations["actor"].enable_corruption = False
    cmdcfg.reset_pose_noise = {}
    cmdcfg.reset_velocity_noise = {}
    cmdcfg.reset_joint_position_noise = (0., 0.)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    agentcfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agentcfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agentcfg), device=args.device)
    infos = runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    validate_checkpoint_motion(infos, args.motion, identity)
    policy = runner.get_inference_policy(device=args.device)
    cmd = env.command_manager.get_term("reference_trajectory")
    starts = torch.tensor(args.start_times, device=args.device).mul(cmd.reference.fps).round().long()
    if torch.any(starts >= cmd.reference.num_frames - 1):
      parser.error("Start time must precede the final reference frame.")
    original_sample = cmd._sample_start_frames

    def sample_fixed(self, env_ids):
      self.frame_ids[env_ids] = starts[env_ids]

    cmd._sample_start_frames = MethodType(sample_fixed, cmd)
    wrapped.reset()
    cmd._sample_start_frames = original_sample
    # Reset forwards physics but does not refresh aligned reference caches.
    cmd.frame_ids -= 1
    cmd._update_command()
    obs = wrapped.get_observations()
    records = {}
    terms = env.termination_manager
    original_compute = terms.compute
    end_names = cfg.terminations["reference_body_height"].params["body_names"]
    body_ids = [cmd.cfg.body_names.index(n) for n in end_names]
    active = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)
    body_error_sum = torch.zeros(env.num_envs, device=args.device)
    anchor_error_sum = torch.zeros_like(body_error_sum)
    measured_steps = torch.zeros_like(body_error_sum)

    def compute_record():
      done = original_compute()
      clip_end = cmd.frame_ids >= cmd.reference.num_frames - 1
      body_error_sum.add_((cmd.aligned_body_position_w - cmd.robot_body_pos_w).norm(dim=-1).mean(dim=-1) * active)
      anchor_error_sum.add_((cmd.anchor_pos_w - cmd.robot_anchor_pos_w).norm(dim=-1) * active)
      measured_steps.add_(active)
      for i in (done | clip_end).nonzero().flatten().tolist():
        if i in records:
          continue
        errors = (cmd.aligned_body_position_w[i, body_ids, 2]
                  - cmd.robot_body_pos_w[i, body_ids, 2]).abs()
        failed = bool(terms.terminated[i])
        records[i] = {
          "start_s": float(starts[i]) / cmd.reference.fps,
          "duration_s": float(env.episode_length_buf[i]) * env.step_dt,
          "reference_s": float(cmd.frame_ids[i]) / cmd.reference.fps,
          "failed": failed,
          "reached_time_limit": bool(terms.time_outs[i]) and not failed,
          "reached_reference_end": bool(clip_end[i]) and not failed,
          "terminations": [n for n in terms.active_terms if bool(terms.get_term(n)[i])],
          "height_errors_m": dict(zip(end_names, errors.tolist())),
          "mean_body_error_m": float(body_error_sum[i] / measured_steps[i]),
          "mean_anchor_error_m": float(anchor_error_sum[i] / measured_steps[i]),
        }
        active[i] = False
      return done

    terms.compute = compute_record
    with torch.inference_mode():
      for _ in range(env.max_episode_length + 1):
        obs, _, _, _ = wrapped.step(policy(obs, stochastic_output=False))
        if len(records) == env.num_envs:
          break
    rows = [records[i] for i in sorted(records)]
    if len(rows) != env.num_envs:
      raise RuntimeError(f"Incomplete evaluation: recorded {len(rows)}/{env.num_envs} attempts.")
    summary = {
      "motion": args.motion, "motion_identity": identity,
      "checkpoint": str(args.checkpoint.resolve()), "condition": args.condition,
      "seed": args.seed, "episode_cap_s": args.duration,
      "episodes": len(rows),
      "full_motion_from_zero": args.full_motion,
      "deterministic": True,
      "noise_scale": args.noise_scale if args.condition == "train" else 0.0,
      "push_scale": args.push_scale if args.condition == "train" else 0.0,
      "mean_body_error_m": sum(r["mean_body_error_m"] for r in rows) / len(rows),
      "mean_anchor_error_m": sum(r["mean_anchor_error_m"] for r in rows) / len(rows),
      "mean_duration_s": sum(r["duration_s"] for r in rows) / len(rows),
      "time_limit_successes": sum(r["reached_time_limit"] for r in rows),
      "reference_end_successes": sum(r["reached_reference_end"] for r in rows),
      "failures": sum(r["failed"] for r in rows),
      "termination_counts": dict(Counter(t for r in rows for t in r["terminations"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
