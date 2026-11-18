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

"""Interactive playback using the original Blink policy and tracking simulator."""

import argparse
from pathlib import Path

import torch

from .runtime import prepare


class BlinkMimicPlayback:
    """Interactive playback with an environment reset on motion selection."""

    def __init__(self, checkpoint_path, motion_dir, robot, device="cuda:0"):
        prepare()
        from blinkmimic import networks
        from blinkmimic.dr import XiLayout
        from blinkmimic.env import TrackingSim, make_sim
        from blinkmimic.kinematics import Kinematics
        from blinkmimic.mdp import MDP
        from .export import load_policy
        from .motion import ExactMotionLib
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.policy = load_policy(checkpoint, robot).policy.to(device).eval()
        networks.AMP_DTYPE = (
            torch.bfloat16 if checkpoint["args"].get("amp", "none") == "bf16" else None
        )
        self.future_frames = checkpoint["args"]["goal_frames"]
        kin = Kinematics(robot, device)
        self.motion = ExactMotionLib.from_dir(motion_dir, robot, kin, device)
        mdp = MDP(kin, self.motion, torch.tensor(robot.default_joint_pos_list(), device=device),
                  XiLayout(robot, device).dim, robot.sensing)
        self.env = TrackingSim(make_sim(robot), 1, robot, self.motion, mdp, device,
                               eval_mode=True, randomize=False)
        self.clip_index = 0
        self.history = None
        self.is_finished = False
        self.has_failed = False
        self.reset()

    @torch.no_grad()
    def select_motion(self, clip_index):
        if not 0 <= clip_index < len(self.motion.names):
            raise ValueError(f"Invalid clip index: {clip_index}")
        self.clip_index = clip_index
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.env.f.copy_(self.motion.clip_start[self.clip_index].reshape(1))
        self.env.done_eval.zero_()
        self.env.failed_eval.zero_()
        self.is_finished = False
        self.has_failed = False

        ids = torch.zeros(1, dtype=torch.long, device=self.env.device)
        self.env.reset(ids, self.env.f.clone())
        self.history = None

    @torch.no_grad()
    def step(self):
        if self.is_finished:
            self.select_motion(self.clip_index)
        observation = self.env.observe().unsqueeze(1)
        self.history = (
            observation.repeat(1, self.policy.hist, 1) if self.history is None
            else torch.cat((self.history[:, 1:], observation), dim=1)
        )
        action = self.policy(self.history, self.motion.goal(self.env.f, self.future_frames))
        if not torch.isfinite(action).all():
            raise ValueError("Checkpoint produced nonfinite actions")
        self.env.step(action)
        self.is_finished = bool(self.env.done_eval[0])
        self.has_failed = bool(self.env.failed_eval[0])


def main(argv=None, *, robot=None):
    from .config import TASKS

    parser = argparse.ArgumentParser(description=__doc__)
    if robot is None:
        parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-steps", type=int)
    args = parser.parse_args(argv)
    from .export import export_policy

    if robot is None:
        prepare()
        robot = TASKS[args.task].robot_factory()

    path = export_policy(args.checkpoint, args.checkpoint.parent, robot, overwrite=True)
    print(f"[INFO] Exported policy: {path}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    playback = BlinkMimicPlayback(args.checkpoint, args.motion_dir, robot)
    from .viewer import run_viewer

    run_viewer(playback, args.checkpoint.name, port=args.port, num_steps=args.num_steps)


if __name__ == "__main__":
    main()
