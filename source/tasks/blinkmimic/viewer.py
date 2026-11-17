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

"""Browser controls and rendering for a single BlinkMimic simulation."""

import queue
import time

import mujoco
import viser
from mjviser import ViserMujocoScene


def run_viewer(playback, checkpoint_name, *, port=8080, num_steps=None):
    from blinkmimic.motion import CONTROL_DT

    server = viser.ViserServer(host="127.0.0.1", port=port, label="BlinkMimic")
    model = playback.env.sim.mj_model
    data = mujoco.MjData(model)
    commands = queue.SimpleQueue()
    try:
        scene = ViserMujocoScene(server, model, num_envs=1)
        server.gui.add_markdown(f"**BlinkMimic** — `{checkpoint_name}`")
        clip = server.gui.add_dropdown("Motion", options=tuple(playback.motion.names))
        is_playing = server.gui.add_checkbox("Play", initial_value=True)
        restart = server.gui.add_button("Restart motion")
        reset = server.gui.add_button("Reset environment")
        status = server.gui.add_markdown("")

        @clip.on_update
        def select_clip(_):
            commands.put(playback.motion.names.index(clip.value))

        @restart.on_click
        def restart_motion(_):
            commands.put(playback.motion.names.index(clip.value))

        @reset.on_click
        def reset_environment(_):
            commands.put(None)

        @server.on_client_connect
        def set_camera(client):
            client.camera.position = (2.0, -2.5, 1.5)
            client.camera.look_at = (0.0, 0.0, 0.65)
            client.camera.up_direction = (0.0, 0.0, 1.0)

        def render():
            data.qpos[:] = playback.env.sim.data.qpos[0].cpu().numpy()
            mujoco.mj_forward(model, data)
            scene.update_from_mjdata(data)

        render()
        print(f"[INFO] BlinkMimic GUI ready: http://localhost:{server.get_port()}", flush=True)
        steps = 0
        deadline = time.monotonic()
        while num_steps is None or steps < num_steps:
            changed = False
            while not commands.empty():
                clip_index = commands.get_nowait()
                if clip_index is None:
                    playback.reset()
                else:
                    playback.select_motion(clip_index)
                changed = True
            if is_playing.value:
                playback.step()
                steps += 1
                changed = True
                if playback.has_failed:
                    is_playing.value = False
            if changed:
                render()
            if steps % 10 == 0 or playback.is_finished:
                frame = int(playback.env.f[0] - playback.motion.clip_start[playback.clip_index])
                outcome = "Failed — paused" if playback.has_failed else "Playing"
                if not is_playing.value and not playback.has_failed:
                    outcome = "Paused"
                status.content = f"{playback.motion.names[playback.clip_index]} · frame {frame} · {outcome}"
            deadline += CONTROL_DT
            now = time.monotonic()
            time.sleep(max(0.0, deadline - now))
            deadline = max(deadline, now)
    finally:
        server.stop()
