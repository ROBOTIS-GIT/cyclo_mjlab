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

"""Adapt SuperMimic's ghost rendering calls to the installed mjlab viewer."""


from supermimic.scripts.viewers import SuperMimicViserViewer


class ClipPlaybackViewer(SuperMimicViserViewer):
    def _apply_selected_clip(self):
        command = self.env.unwrapped.command_manager.get_term("motion")
        index = self._selected_clip_idx
        command.playback_clip_start = 0 if index == 0 else int(command.motion.motion_ends[index - 1])
        super()._apply_selected_clip()


class _GhostVisualizer:
    def __init__(
        self, visualizer, show_reference=True, show_prediction=True, *,
        lateral_index=1, spacing=0.0,
    ):
        self._visualizer = visualizer
        self._show_reference = show_reference
        self._show_prediction = show_prediction
        self._lateral_index = lateral_index
        self._spacing = spacing

    def __getattr__(self, name):
        return getattr(self._visualizer, name)

    def add_ghost_entity(self, qpos, **kwargs):
        if self._show_prediction:
            return self._draw(qpos, self._spacing, **kwargs)

    def add_ghost_mesh(self, qpos, **kwargs):
        if self._show_reference:
            return self._draw(qpos, -self._spacing, **kwargs)

    def _draw(self, qpos, offset, **kwargs):
        pose = qpos.copy()
        pose[self._lateral_index] += offset
        kwargs["alpha"] = 0.8
        return self._visualizer.add_ghost_mesh(pose, **kwargs)


def adapt_ghost_visualizer(viewer):
    """Keep the upstream prediction callback while adapting its scene API."""
    env = viewer.env.unwrapped
    update_visualizers = env.update_visualizers
    motion = env.command_manager.get_term("motion")
    # This adapter owns the reference switch; omit the second Motion debug checkbox.
    motion.cfg.debug_vis = False
    motion._ghost_color[:] = (0.1, 0.85, 0.2, 0.8)
    motion._ghost_model = None
    lateral_index = int(env.scene["robot"].indexing.free_joint_q_adr[1])
    model = viewer.wm_vis_model
    model.geom_rgba[:, :3] = (0.05, 0.35, 1.0)
    for geom in range(model.ngeom):
        if model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0:
            model.geom_rgba[geom, 3] = 0.0

    settings = {"reference": True, "prediction": True, "separate": True}
    original_debug_gui = env.command_manager.create_debug_vis_gui

    def create_debug_gui(server, on_change=None):
        original_debug_gui(server, on_change=on_change)
        for key, label in (
            ("reference", "Reference Motion (green)"),
            ("prediction", "World Model Prediction (blue)"),
            ("separate", "Separate Ghosts"),
        ):
            checkbox = server.gui.add_checkbox(label, initial_value=settings[key])

            def changed(event, setting=key):
                settings[setting] = event.target.value
                if on_change is not None:
                    on_change()

            checkbox.on_update(changed)
        server.gui.add_markdown("Separation shifts display only; simulation coordinates are unchanged.")

    env.command_manager.create_debug_vis_gui = create_debug_gui

    def update(visualizer):
        adapted = _GhostVisualizer(
            visualizer, settings["reference"], settings["prediction"],
            lateral_index=lateral_index, spacing=0.9 if settings["separate"] else 0.0,
        )
        update_visualizers(adapted)
        if settings["reference"]:
            motion._debug_vis_impl(adapted)

    env.update_visualizers = update
