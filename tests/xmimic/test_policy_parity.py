"""Compare the real JAX EMA checkpoint format with the Cyclo ONNX export."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from source.tasks.blinkmimic.runtime import prepare
from source.tasks.xmimic.config import TASKS
from source.tasks.xmimic.export import export_policy
from source.tasks.xmimic.train import UPSTREAM


def test_jax_checkpoint_matches_onnx(tmp_path):
    python = os.environ.get("XMIMIC_TEST_PYTHON")
    if not python:
        pytest.skip("Set XMIMIC_TEST_PYTHON to the separate JAX interpreter")
    ort = pytest.importorskip("onnxruntime")
    prepare()
    robot = next(iter(TASKS.values())).robot_factory()
    metadata = tmp_path / "meta.json"
    from json import dumps

    from source.tasks.xmimic.config import robot_contract

    metadata.write_text(dumps(robot_contract(robot)))
    checkpoint = tmp_path / "checkpoint.npz"
    subprocess.run(
        [
            str(Path(python).absolute()),
            "-c",
            """
import json
import sys

import jax
import jax.numpy as jnp
from xmimic import nets
from xmimic.runtime import export_policy

robot = json.load(open(sys.argv[1]))
j = len(robot["joint_names"])
h, k = 10, 10
obs_dim, goal_dim = 9 + 3*j + 3, k*(15 + 2*j)
params, bufs = nets.pi_init(jax.random.PRNGKey(0), obs_dim, goal_dim, j, h)
for norm in bufs.values():
    norm["mean"] = jnp.linspace(-0.3, 0.2, norm["mean"].size)
    norm["var"] = jnp.linspace(0.25, 1.25, norm["var"].size)
ohn = jnp.linspace(-1, 1, 3*h*obs_dim).reshape(3, h, obs_dim)
goal = jnp.linspace(-0.3, 0.3, 3*goal_dim).reshape(3, goal_dim)
ref = (15, jnp.array(robot["default_joint_pos"]), jnp.array(robot["action_scale"]))
nets.MM_OUT = jnp.bfloat16
action = nets.pi_apply(params, bufs, ohn, goal, ref)
args = dict(robot=robot, hist=h, goal_frames=k, enc_hidden="2048,1024,512,256",
            dec_hidden="2048,1024,512,256", pi_ema=0.995)
export_policy(sys.argv[2], (params, bufs), {"args": args}, (ohn, goal, action))
""",
            str(metadata),
            str(checkpoint),
        ],
        cwd=UPSTREAM,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=True,
    )
    path = export_policy(checkpoint, tmp_path / "bundle", robot)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    with np.load(checkpoint, allow_pickle=False) as data:
        inputs = np.concatenate(
            (data["test_oh"].reshape(3, -1), data["test_g"]), axis=1
        )
        actual = session.run(["actions"], {"observations": inputs})[0]
        np.testing.assert_allclose(actual, data["test_a"], rtol=0.01, atol=0.003)
    assert (path.parents[1] / "params/sim2real.yaml").is_file()
