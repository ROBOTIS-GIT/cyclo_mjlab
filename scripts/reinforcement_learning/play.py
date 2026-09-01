# Copyright 2026 ROBOTIS CO., LTD.
# Copyright 2025, The mjlab Developers
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
# Author: Insu Park
#
# This file includes modifications by ROBOTIS CO., LTD. to code derived
# from mujocolab/mjlab.

"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Prefer task and asset modules from this checkout over an installed copy.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from source.tasks.mimic.mdp import ReferenceTrajectoryCommandCfg


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  motion_file: str | None = None
  reference_ghost_offset: tuple[float, float, float] | None = None
  """World-frame offset for the Mimic reference ghost."""
  num_envs: int | None = None
  num_steps: int | None = None
  """Stop after this many policy steps. None keeps the viewer running."""
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 1
  """Record one frame every N policy steps."""
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer_world: bool = False
  """Use a fixed world-frame camera instead of tracking one robot."""
  viewer_lookat: tuple[float, float, float] | None = None
  """World point the viewer camera looks at."""
  viewer_distance: float | None = None
  viewer_elevation: float | None = None
  viewer_azimuth: float | None = None
  viewer_end_lookat: tuple[float, float, float] | None = None
  """Final look-at point for a smooth video camera move."""
  viewer_end_distance: float | None = None
  viewer_end_elevation: float | None = None
  viewer_end_azimuth: float | None = None
  viewer_follow_env: int | None = None
  """Environment whose robot is followed; -1 selects a fast central robot."""
  viewer_follow_body: str = "torso_link"
  viewer_follow_start: float = 0.65
  """Normalized video progress at which following starts."""
  viewer_max_extra_envs: int | None = None
  """Maximum number of neighboring environments included in offscreen video."""
  viewer_shadows: bool = True
  """Enable shadows in offscreen video rendering."""
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  export_policy: bool = True
  """Export policy.onnx and policy.pt."""
  export_only: bool = False
  """Export policy files and exit without starting a viewer."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  if cfg.video_interval < 1:
    raise ValueError("--video-interval must be at least 1.")
  if not 0.0 <= cfg.viewer_follow_start < 1.0:
    raise ValueError("--viewer-follow-start must be in the range [0, 1).")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  if cfg.export_only and (DUMMY_MODE or not cfg.export_policy):
    raise ValueError("--export-only requires --agent trained and --export-policy True.")

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for reference trajectory command.
  is_tracking_task = "reference_trajectory" in env_cfg.commands and isinstance(
    env_cfg.commands["reference_trajectory"], ReferenceTrajectoryCommandCfg
  )

  if is_tracking_task:
    motion_cmd = env_cfg.commands["reference_trajectory"]
    assert isinstance(motion_cmd, ReferenceTrajectoryCommandCfg)

    selected_motion_file = cfg.motion_file or motion_cmd.trajectory_file
    if not selected_motion_file:
      raise ValueError(
        "Mimic tasks require a trajectory file in the task configuration "
        "or through --motion-file."
      )
    motion_path = Path(selected_motion_file).expanduser().resolve()
    if not motion_path.is_file():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd.trajectory_file = str(motion_path)
    if cfg.reference_ghost_offset is not None:
      motion_cmd.debug_vis = True
      motion_cmd.viz.mode = "ghost"
      motion_cmd.viz.ghost_offset = cfg.reference_ghost_offset
      print(f"[INFO]: Reference ghost offset: {cfg.reference_ghost_offset}")
    print(f"[INFO]: Using motion file: {motion_cmd.trajectory_file}")
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    elif cfg.wandb_run_path is not None:
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    else:
      resume_path = get_checkpoint_path(
        log_root_path,
        run_dir=r".*",
        checkpoint=r"model_\d+\.pt",
      )
      print(f"[INFO]: Auto-selected latest checkpoint: {resume_path}")
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width
  if cfg.viewer_world:
    env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.WORLD
  if cfg.viewer_lookat is not None:
    env_cfg.viewer.lookat = cfg.viewer_lookat
  if cfg.viewer_distance is not None:
    env_cfg.viewer.distance = cfg.viewer_distance
  if cfg.viewer_elevation is not None:
    env_cfg.viewer.elevation = cfg.viewer_elevation
  if cfg.viewer_azimuth is not None:
    env_cfg.viewer.azimuth = cfg.viewer_azimuth
  if cfg.viewer_max_extra_envs is not None:
    env_cfg.viewer.max_extra_envs = cfg.viewer_max_extra_envs
  env_cfg.viewer.enable_shadows = cfg.viewer_shadows

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    has_base_velocity_command = "base_velocity" in env_cfg.commands

    follow_robot = None
    follow_body_index = None
    follow_env_index = (
      None if cfg.viewer_follow_env == -1 else cfg.viewer_follow_env
    )
    central_env_ids = None
    if cfg.viewer_follow_env is not None:
      if not cfg.viewer_world:
        raise ValueError("--viewer-follow-env requires --viewer-world True.")
      if not -1 <= cfg.viewer_follow_env < env.num_envs:
        raise ValueError(
          f"--viewer-follow-env must be -1 or between 0 and {env.num_envs - 1}."
        )
      if cfg.viewer_follow_env == -1:
        central_count = min(9, env.num_envs)
        origin_distance = torch.sum(env.scene.env_origins[:, :2] ** 2, dim=1)
        central_env_ids = torch.topk(
          origin_distance, central_count, largest=False
        ).indices
      follow_robot = env.scene["robot"]
      body_indices, _ = follow_robot.find_bodies(cfg.viewer_follow_body)
      if not body_indices:
        raise ValueError(
          f"Body '{cfg.viewer_follow_body}' was not found in the robot."
        )
      follow_body_index = body_indices[0]

    camera_motion_enabled = cfg.viewer_follow_env is not None or any(
      value is not None
      for value in (
        cfg.viewer_end_lookat,
        cfg.viewer_end_distance,
        cfg.viewer_end_elevation,
        cfg.viewer_end_azimuth,
      )
    )
    start_lookat = tuple(env_cfg.viewer.lookat)
    start_distance = env_cfg.viewer.distance
    start_elevation = env_cfg.viewer.elevation
    start_azimuth = env_cfg.viewer.azimuth
    end_lookat = (
      cfg.viewer_end_lookat
      if cfg.viewer_end_lookat is not None
      else start_lookat
    )
    end_distance = (
      cfg.viewer_end_distance
      if cfg.viewer_end_distance is not None
      else start_distance
    )
    end_elevation = (
      cfg.viewer_end_elevation
      if cfg.viewer_end_elevation is not None
      else start_elevation
    )
    end_azimuth = (
      cfg.viewer_end_azimuth
      if cfg.viewer_end_azimuth is not None
      else start_azimuth
    )

    class IntervalVideoRecorder(VideoRecorder):
      """Record interval frames with an optional smooth camera move."""

      def _record_frame(self) -> None:
        if self.step_count % cfg.video_interval == 0:
          if camera_motion_enabled:
            self._update_camera()
          super()._record_frame()

      def _update_camera(self) -> None:
        nonlocal follow_env_index

        renderer = self._wrapped_env._offline_renderer
        if renderer is None:
          return

        if self.video_length is None or self.video_length <= 1:
          raw_progress = 1.0
        else:
          raw_progress = min(
            len(self.current_video_frames) / (self.video_length - 1), 1.0
          )
        progress = raw_progress
        progress = progress * progress * (3.0 - 2.0 * progress)

        def interpolate(start: float, end: float) -> float:
          return start + (end - start) * progress

        lookat = tuple(
          interpolate(start, end)
          for start, end in zip(start_lookat, end_lookat, strict=True)
        )
        if (
          cfg.viewer_follow_env == -1
          and follow_env_index is None
          and raw_progress >= cfg.viewer_follow_start
        ):
          assert central_env_ids is not None
          if has_base_velocity_command:
            velocity_commands = self._wrapped_env.command_manager.get_command(
              "base_velocity"
            )
            movement_score = torch.linalg.vector_norm(
              velocity_commands[central_env_ids, :2], dim=1
            ) + 0.25 * torch.abs(velocity_commands[central_env_ids, 2])
          else:
            assert follow_robot is not None
            movement_score = torch.linalg.vector_norm(
              follow_robot.data.root_link_lin_vel_w[central_env_ids, :2], dim=1
            )
          follow_env_index = int(
            central_env_ids[torch.argmax(movement_score)].item()
          )
          if has_base_velocity_command:
            selected_command = velocity_commands[follow_env_index].tolist()
            print(
              f"[INFO] Auto-selected follow env {follow_env_index} with command "
              f"({selected_command[0]:.2f}, {selected_command[1]:.2f}, "
              f"{selected_command[2]:.2f})"
            )
          else:
            selected_speed = float(torch.max(movement_score).item())
            print(
              f"[INFO] Auto-selected follow env {follow_env_index} with planar "
              f"speed {selected_speed:.2f} m/s"
            )
        if (
          follow_env_index is not None
          and follow_robot is not None
          and follow_body_index is not None
          and raw_progress >= cfg.viewer_follow_start
        ):
          follow_progress = (raw_progress - cfg.viewer_follow_start) / (
            1.0 - cfg.viewer_follow_start
          )
          follow_progress = follow_progress * follow_progress * (
            3.0 - 2.0 * follow_progress
          )
          body_position = (
            follow_robot.data.body_link_pos_w[
              follow_env_index, follow_body_index
            ]
            .detach()
            .cpu()
            .tolist()
          )
          lookat = tuple(
            start + (end - start) * follow_progress
            for start, end in zip(lookat, body_position, strict=True)
          )

        renderer._cam.lookat[:] = lookat
        renderer._cam.distance = interpolate(start_distance, end_distance)
        renderer._cam.elevation = interpolate(start_elevation, end_elevation)
        renderer._cam.azimuth = interpolate(start_azimuth, end_azimuth)

    if cfg.video_interval > 1:
      env.metadata = dict(env.metadata)
      env.metadata["render_fps"] = max(
        1, round(env.metadata["render_fps"] / cfg.video_interval)
      )
      print(
        f"[INFO] Recording every {cfg.video_interval} policy steps "
        f"at {env.metadata['render_fps']} fps"
      )
    if camera_motion_enabled:
      print(
        "[INFO] Animating viewer camera: "
        f"distance {start_distance:.1f}->{end_distance:.1f}, "
        f"elevation {start_elevation:.1f}->{end_elevation:.1f}, "
        f"azimuth {start_azimuth:.1f}->{end_azimuth:.1f}"
      )
    if cfg.viewer_follow_env == -1:
      print(
        f"[INFO] A fast central robot will be selected at "
        f"{cfg.viewer_follow_start:.0%} progress"
      )
    elif follow_env_index is not None:
      print(
        f"[INFO] Following env {follow_env_index} body "
        f"'{cfg.viewer_follow_body}' from {cfg.viewer_follow_start:.0%} progress"
      )

    env = IntervalVideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    if cfg.export_policy:
      assert log_dir is not None
      manager_env = env.unwrapped
      export_model_dir = log_dir / "exported"
      export_model_dir.mkdir(parents=True, exist_ok=True)
      onnx_path = export_model_dir / "policy.onnx"
      jit_path = export_model_dir / "policy.pt"
      runner.export_policy_to_onnx(str(export_model_dir), onnx_path.name)
      runner.export_policy_to_jit(str(export_model_dir), jit_path.name)
      metadata = get_base_metadata(manager_env, log_dir.name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      print(f"[INFO] Exported ONNX policy: {onnx_path}")
      print(f"[INFO] Exported TorchScript policy: {jit_path}")

    if cfg.export_only:
      env.close()
      return

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run(num_steps=cfg.num_steps)
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run(num_steps=cfg.num_steps)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import source.tasks

  k1_task_prefixes = ("Cyclo-Velocity-", "Cyclo-Mimic-K1-Rev1-")
  all_tasks = [
    task_id for task_id in list_tasks() if task_id.startswith(k1_task_prefixes)
  ]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
