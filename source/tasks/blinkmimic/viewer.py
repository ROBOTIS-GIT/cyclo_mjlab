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

"""Browser controls and mjlab scene rendering for BlinkMimic playback."""

import copy
import queue
import time

import mujoco
import numpy as np
import viser
from mjlab.viewer.viser.scene import MjlabViserScene


class BlinkMimicViewer:
    def __init__(self, playback, checkpoint_name, *, port=8080):
        self.playback = playback
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label="BlinkMimic")
        self.commands = queue.SimpleQueue()
        self.is_paused = False
        self.speed = 1.0
        self.steps = 0
        self.checkpoint_name = checkpoint_name

    def setup(self):
        sim = self.playback.env.sim
        self.data = mujoco.MjData(sim.mj_model)
        self.scene = MjlabViserScene(
            self.server, sim.mj_model, num_envs=1,
            sim_model=sim.model, expanded_fields=sim.expanded_fields,
        )
        self.scene.camera_tracking_enabled = True
        self.scene.debug_visualization_enabled = True
        self.reference_model = copy.deepcopy(sim.mj_model)
        for geom in range(self.reference_model.ngeom):
            is_collision = (self.reference_model.geom_contype[geom] != 0 or
                            self.reference_model.geom_conaffinity[geom] != 0)
            self.reference_model.geom_rgba[geom] = (0.1, 0.85, 0.2, 0.0 if is_collision else 0.8)
        self.scene.set_refresh_handler(lambda: self.commands.put(("render", None)))
        tabs = self.server.gui.add_tab_group()
        with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
            with self.server.gui.add_folder("Info"):
                self.server.gui.add_markdown(f"**BlinkMimic** — `{self.checkpoint_name}`")
                self.status = self.server.gui.add_markdown("")
            with self.server.gui.add_folder("Simulation"):
                self.pause_button = self.server.gui.add_button(
                    "Pause", icon=viser.Icon.PLAYER_PAUSE,
                )
                self.pause_button.on_click(lambda _: self.commands.put(("pause", None)))
                step = self.server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)
                step.on_click(lambda _: self.commands.put(("step", None)))
                reset = self.server.gui.add_button("Reset Environment")
                reset.on_click(lambda _: self.commands.put(("reset", None)))
                speed = self.server.gui.add_button_group(
                    "Speed", options=["Slower", "1x", "Faster"],
                )
                speed.on_click(lambda event: self.commands.put(("speed", event.target.value)))
            with self.server.gui.add_folder("Commands"):
                self.clip = self.server.gui.add_dropdown(
                    "Motion Clip", options=tuple(self.playback.motion.names),
                    initial_value=self.playback.motion.names[self.playback.clip_index],
                )
                self.clip.on_update(lambda event: self.commands.put(("motion", event.target.value)))
                restart = self.server.gui.add_button("Restart Motion")
                restart.on_click(lambda _: self.commands.put(("motion", self.clip.value)))
            with self.server.gui.add_folder("Scene"):
                self.scene.create_scene_gui(debug_viz_extra_gui=self.create_ghost_controls)
        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            self.scene.create_overlay_gui()
        self.render()

    def create_ghost_controls(self):
        self.reference_toggle = self.server.gui.add_checkbox("Reference Motion (green)", initial_value=True)
        self.reference_toggle.on_update(lambda _: self.commands.put(("render", None)))
        self.separate_reference = self.server.gui.add_checkbox("Separate Ghost", initial_value=True)
        self.separate_reference.on_update(lambda _: self.commands.put(("render", None)))

    def set_paused(self, is_paused):
        self.is_paused = is_paused
        self.pause_button.label = "Play" if is_paused else "Pause"
        self.pause_button.icon = viser.Icon.PLAYER_PLAY if is_paused else viser.Icon.PLAYER_PAUSE

    def step(self):
        self.playback.step()
        self.steps += 1
        if self.playback.has_failed:
            self.set_paused(True)

    def process_commands(self):
        changed = False
        while not self.commands.empty():
            action, value = self.commands.get_nowait()
            if action == "pause":
                self.set_paused(not self.is_paused)
            elif action == "step":
                self.set_paused(True)
                self.step()
            elif action == "reset":
                self.playback.reset()
            elif action == "motion":
                self.playback.select_motion(self.playback.motion.names.index(value))
            elif action == "speed":
                if value == "1x":
                    self.speed = 1.0
                else:
                    factor = 0.5 if value == "Slower" else 2.0
                    self.speed = min(4.0, max(0.125, self.speed * factor))
            changed = True
        return changed

    def reference_qpos(self):
        motion = self.playback.motion
        frame = min(int(self.playback.env.f[0]), int(motion.clip_end[self.playback.clip_index]))
        indexing = self.playback.env.robot.indexing
        root_indices = indexing.free_joint_q_adr.cpu().numpy()
        joint_indices = indexing.joint_q_adr.cpu().numpy()
        qpos = self.data.qpos.copy()
        qpos[root_indices[:3]] = (
            motion.root_p[frame] + self.playback.env.scene.env_origins[0]
        ).cpu().numpy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, motion.root_R[frame].cpu().numpy().astype(float).reshape(9))
        qpos[root_indices[3:7]] = quat
        qpos[joint_indices] = motion.q[frame].cpu().numpy()
        if self.separate_reference.value:
            qpos[root_indices[1]] -= 0.9
        return qpos

    def render(self):
        sim = self.playback.env.sim
        self.data.qpos[:] = sim.data.qpos[0].cpu().numpy()
        self.data.qvel[:] = sim.data.qvel[0].cpu().numpy()
        self.data.ctrl[:] = sim.data.ctrl[0].cpu().numpy()
        mujoco.mj_forward(sim.mj_model, self.data)
        self.scene.paused = self.is_paused
        self.scene.clear()
        if self.reference_toggle.value:
            self.scene.add_ghost_mesh(
                self.reference_qpos(), self.reference_model, alpha=0.8, label="reference_motion",
            )
        self.scene.update_from_mjdata(self.data)
        frame = int(self.playback.env.f[0] - self.playback.motion.clip_start[self.playback.clip_index])
        outcome = "Paused" if self.is_paused else "Playing"
        if self.playback.has_failed:
            outcome = "Failed — paused"
        self.status.content = (
            f"{self.playback.motion.names[self.playback.clip_index]} · frame {frame}\n\n"
            f"{outcome} · {self.speed:g}x"
        )

    def run(self, num_steps=None):
        from blinkmimic.motion import CONTROL_DT

        try:
            self.setup()
            print(f"[INFO] BlinkMimic GUI ready: http://localhost:{self.server.get_port()}", flush=True)
            deadline = time.monotonic()
            while num_steps is None or self.steps < num_steps:
                changed = self.process_commands()
                if not self.is_paused and (num_steps is None or self.steps < num_steps):
                    self.step()
                    changed = True
                if changed:
                    self.render()
                # Speed changes wall-clock pacing, never the policy or physics timestep.
                deadline += CONTROL_DT / self.speed if not self.is_paused else CONTROL_DT
                now = time.monotonic()
                time.sleep(max(0.0, deadline - now))
                deadline = max(deadline, now)
        finally:
            self.server.stop()


def run_viewer(playback, checkpoint_name, *, port=8080, num_steps=None):
    BlinkMimicViewer(playback, checkpoint_name, port=port).run(num_steps)
