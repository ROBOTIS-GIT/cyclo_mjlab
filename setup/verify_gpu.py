"""Check both GPUs and headless rendering in the installed container."""
import importlib.metadata as md
import json

import mujoco
import torch
import warp as wp

versions = {name: md.version(name) for name in (
    "torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib"
)}
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
wp.init()
gpus = []
for index in range(torch.cuda.device_count()):
    device = f"cuda:{index}"
    with torch.cuda.device(index):
        a = torch.ones((256, 256), device=device)
        result = a @ a
        torch.cuda.synchronize(index)
        assert torch.all(result == 256).item()
        w = wp.zeros(16, dtype=wp.float32, device=device)
        wp.synchronize_device(device)
        assert w.numpy().sum() == 0
    gpus.append({"index": index, "name": torch.cuda.get_device_name(index),
                 "capability": torch.cuda.get_device_capability(index)})

model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
  <light pos="0 0 3"/><geom type="plane" size="2 2 .1"/>
  <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
</worldbody></mujoco>''')
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    pixels = renderer.render()
    assert pixels.shape == (64, 64, 3)
    assert pixels.max() > 0

print(json.dumps({"versions": versions, "cuda": torch.version.cuda,
                  "gpus": gpus, "egl_render": "passed"}, indent=2))
