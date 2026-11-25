"""Synthetic full-width policy export; no dataset, trained checkpoint or simulator."""

import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from source.tasks.blinkmimic.export import export_policy, load_policy, sha256
from source.tasks.blinkmimic.runtime import ROOT, VENDOR, prepare

prepare()
from blinkmimic.networks import Policy, policy_kwargs  # noqa: E402

from source.tasks.blinkmimic.config.k1_rev1.robot_spec import make_robot  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_policy():
    torch.manual_seed(21)
    robot = make_robot()
    args = dict(
        robot="k1",
        hist=10,
        goal_frames=10,
        enc_hidden="2048,1024,512,256",
        dec_hidden="2048,1024,512,256",
        pi_ref_skip=True,
        pi_ema=0.99,
        amp="bf16",
    )
    policy = Policy(81, 610, 23, **policy_kwargs(args, robot)).eval()
    with torch.no_grad():
        for norm in (policy.obs_norm, policy.goal_norm):
            norm.mean.copy_(torch.linspace(-0.4, 0.8, norm.mean.numel()))
            norm.var.copy_(torch.linspace(0.2, 1.3, norm.var.numel()))
            norm.count.fill_(23456)
        policy.dec[-1].bias.copy_(torch.linspace(-0.2, 0.3, 23))
    return (
        robot,
        policy,
        dict(
            args=args,
            pi=policy.state_dict(),
            pi_live={"must_not_be_loaded": torch.tensor(1.0)},
        ),
    )


@pytest.mark.parametrize("ref_skip", [True, False])
def test_full_width_onnx_matches_original_policy(tmp_path, synthetic_policy, ref_skip):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    robot, original, checkpoint = synthetic_policy
    checkpoint = dict(checkpoint, args=dict(checkpoint["args"], pi_ref_skip=ref_skip))
    if not ref_skip:
        checkpoint["pi"] = {
            k: v
            for k, v in checkpoint["pi"].items()
            if k not in ("ref_q0", "ref_alpha")
        }
        original = copy.deepcopy(original)
        original.ref_off = None
    # A different playback robot must not rewrite the saved policy buffers.
    robot = replace(robot, default_joint_pos={".*": 0.125},
                    action_scale=dict.fromkeys(robot.joint_names, 0.375))
    source = tmp_path / "synthetic.pt"
    torch.save(checkpoint, source)
    destination = tmp_path / "export"
    path = export_policy(source, destination, robot)
    metadata = json.loads((destination / "metadata.json").read_text())
    assert path == destination / "exported/policy.onnx"
    assert not (destination / "params/sim2real.yaml").exists()
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    assert [i.name for i in model.graph.input] == ["observations"]
    assert [o.name for o in model.graph.output] == ["actions"]
    assert metadata["input"]["shape"] == ["batch", 1420]
    assert metadata["weights_key"] == "pi"
    assert metadata["reference_skip"]["enabled"] == ref_skip
    assert not metadata["includes_motion"]
    assert "checkpoint_sha256" not in metadata
    assert metadata["onnx_sha256"] == sha256(path)
    expected_notices = {
        "licenses/Apache-2.0.txt": ROOT / "LICENSE",
        "licenses/BlinkMimic-MIT.txt": VENDOR / "LICENSE",
        "licenses/BlinkMimic-THIRD_PARTY_NOTICES.md": VENDOR / "THIRD_PARTY_NOTICES.md",
    }
    assert set(metadata["software_notices"]) == {"NOTICE.txt", *expected_notices}
    for relative, digest in metadata["software_notices"].items():
        assert sha256(destination / relative) == digest
    for relative, original_license in expected_notices.items():
        assert (destination / relative).read_bytes() == original_license.read_bytes()
    embedded = json.loads(model.metadata_props[0].value)
    assert embedded == {k: v for k, v in metadata.items() if k != "onnx_sha256"}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    ort.disable_telemetry_events()
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    wrapper = load_policy(checkpoint, robot)
    for batch in (1, 3):
        x = torch.randn(batch, 1420) * 0.3
        with torch.no_grad():
            expected = original(x[:, :810].reshape(batch, 10, 81), x[:, 810:])
            torch.testing.assert_close(wrapper(x), expected, rtol=0, atol=0)
        actual = session.run(["actions"], {"observations": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=2e-5)
    with pytest.raises(FileExistsError):
        export_policy(source, destination, robot)
    config = destination / "params/sim2real.yaml"
    config.parent.mkdir()
    config.write_text("training config must remain unchanged\n")
    record = destination / "run-record.json"
    record.write_text("training record must remain unchanged\n")
    assert export_policy(source, destination, robot, overwrite=True) == path
    assert config.read_text() == "training config must remain unchanged\n"
    assert record.read_text() == "training record must remain unchanged\n"
    onnx.checker.check_model(onnx.load(path), full_check=True)


@pytest.mark.parametrize(
    "failure",
    [
        "hist",
        "missing_args",
        "missing_pi",
        "pi_live_only",
        "ref_scale",
        "negative_var",
        "nonfinite",
        "wrong_shape",
        "pruned",
    ],
)
def test_incompatible_checkpoints_are_rejected(synthetic_policy, failure):
    robot, _, source = synthetic_policy
    ck = dict(source, args=dict(source["args"]), pi=dict(source["pi"]))
    if failure == "hist":
        ck["args"]["hist"] = 0
    elif failure == "missing_args":
        del ck["args"]["enc_hidden"]
    elif failure in ("missing_pi", "pi_live_only"):
        del ck["pi"]
    elif failure == "ref_scale":
        ck["pi"]["ref_alpha"] = torch.zeros_like(ck["pi"]["ref_alpha"])
    elif failure == "negative_var":
        ck["pi"]["obs_norm.var"] = -torch.ones(81)
    elif failure == "nonfinite":
        ck["pi"]["obs_norm.mean"] = torch.full((81,), float("nan"))
    elif failure == "wrong_shape":
        ck["args"]["hist"] = 1
    elif failure == "pruned":
        ck["args"]["pi_arch"] = "unsupported"
    with pytest.raises((ValueError, RuntimeError)):
        load_policy(ck, robot)


@pytest.mark.filterwarnings("error::UserWarning")
def test_load_policy_ignores_checkpoint_robot(synthetic_policy):
    robot, _, saved = synthetic_policy
    checkpoint = dict(saved, args=dict(saved["args"], robot="another_robot"))
    load_policy(checkpoint, robot)
