"""Synthetic full-width encoder export; no trained weights or motion datasets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.entity import Entity
from supermimic.representations import create_representation
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor
from supermimic.tasks.task_spec import build_actor_obs_td

from source.learning.export_notices import distribution_license
from source.tasks.supermimic import REPO_ROOT
from source.tasks.supermimic.checkpoint import make_manifest
from source.tasks.supermimic.export import (
    export_policy,
    load_policy_bundle,
    motion_goal,
    nominal_action_contract,
    pack_input,
    sha256,
)
from source.tasks.supermimic.motion import model_joint_names
from source.tasks.supermimic.tasks import TASKS, k1_env_cfg, register_tasks


@pytest.fixture(scope="module")
def synthetic_bundles(tmp_path_factory):
    from mjlab.tasks.registry import list_tasks

    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    torch.manual_seed(42)
    result = {}
    for representation, mode, history in (
        ("gencoord", "proprio", 10),
        ("gencoord", "rep_state", 1),
        ("maxcoord", "proprio", 10),
        ("maxcoord", "rep_state", 1),
    ):
        task = f"Cyclo-SuperMimic-{'GenCoord' if representation == 'gencoord' else 'MaxCoord'}-K1-Rev1"
        cfg = k1_env_cfg(representation)
        cfg.commands["motion"].sampling_mode = "external"
        rcfg = SuperMimicRunnerCfg(
            representation=representation, actor_obs=mode, history_length=history
        )
        model = Entity(cfg.scene.entities["robot"]).spec.compile()
        names = list(model_joint_names(model))
        action = nominal_action_contract(cfg, names)
        rep = create_representation(
            representation,
            59 if representation == "gencoord" else 182,
            23,
            0.02,
            cfg.decimation,
            torch.tensor(action["q0"]),
            actor_obs_mode=mode,
        )
        actor = build_actor(
            rcfg,
            SimpleNamespace(representation=rep, num_actions=23, num_goals=550),
            "cpu",
        ).eval()
        with torch.no_grad():
            for norm in [actor.obs_normalizer, *actor.goal_normalizers.values()]:
                norm._mean.copy_(torch.linspace(-0.2, 0.3, norm._mean.numel())[None])
                norm._std.copy_(torch.linspace(0.4, 1.4, norm._std.numel())[None])
                # Deliberately not std**2: export must preserve saved std, not recompute.
                norm._var.fill_(5)
                norm.count.fill_(12345)
        directory = tmp_path_factory.mktemp(f"synthetic-{representation}-{mode}")
        (directory / "manifest.json").write_text(
            json.dumps(make_manifest(task, cfg, rcfg, []))
        )
        torch.save(actor.state_dict(), directory / "actor.pt")
        torch.save(
            {
                "dynamics": {
                    "action_scale": torch.tensor(action["scale"])[None],
                    "action_offset": torch.tensor(action["offset"])[None],
                }
            },
            directory / "world_model.pt",
        )
        result[representation, mode] = directory, task, actor
    return result


@pytest.mark.parametrize(
    "representation,mode,encoder",
    [
        ("gencoord", "proprio", "motion"),
        ("maxcoord", "proprio", "motion"),
        ("gencoord", "rep_state", "motion"),
        ("maxcoord", "rep_state", "motion"),
        ("gencoord", "proprio", "velocity"),
    ],
)
def test_full_width_motion_policy_onnx_parity(
    tmp_path, synthetic_bundles, representation, mode, encoder
):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    directory, task, original = synthetic_bundles[representation, mode]
    path = export_policy(directory, task, tmp_path / "out", encoder=encoder)
    bundle = load_policy_bundle(directory, task)
    from supermimic.runners.exporter import _SingleEncoderOnnx

    wrapper = _SingleEncoderOnnx(bundle.actor, encoder, bundle.term_widths).eval()
    metadata = json.loads((path.parent / "metadata.json").read_text())
    expected_width = (
        1270 if mode == "proprio" else (576 if representation == "gencoord" else 734)
    )
    if encoder == "velocity":
        expected_width = 753
    assert metadata["input"]["shape"] == [1, expected_width]
    assert metadata["actor_obs"] == mode
    assert metadata["onnx_sha256"] == sha256(path)
    assert set(metadata["software_notices"]) == {
        "NOTICE.txt",
        "licenses/Apache-2.0.txt",
        "licenses/RSL-RL-BSD-3-Clause.txt",
    }
    notice = (path.parent / "NOTICE.txt").read_text()
    assert "Copyright (c) 2020 Preferred Networks, Inc." in notice
    assert "Copyright 2026 The SuperMimic Authors" in notice
    for relative, digest in metadata["software_notices"].items():
        assert sha256(path.parent / relative) == digest
    assert (path.parent / "licenses/Apache-2.0.txt").read_bytes() == (
        REPO_ROOT / "third_party/supermimic/LICENSE"
    ).read_bytes()
    assert (
        path.parent / "licenses/RSL-RL-BSD-3-Clause.txt"
    ).read_bytes() == distribution_license("rsl-rl-lib").read_bytes()
    assert metadata["source_sha256"]["actor.pt"] == sha256(directory / "actor.pt")
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    assert [i.name for i in model.graph.input] == ["obs"]
    assert [o.name for o in model.graph.output] == ["actions"]
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
    original.set_slot_mix({encoder: 1.0})
    for batch in (1, 3):
        history = torch.randn(batch, original.history_length, original.obs_dim)
        goal = torch.randn(batch, 520 if encoder == "motion" else 3)
        flat = pack_input(history, goal, bundle.term_widths)
        with torch.no_grad():
            expected = original(build_actor_obs_td({encoder: goal}, history))
            torch.testing.assert_close(wrapper(flat), expected, rtol=0, atol=0)
        if batch == 1:  # Upstream exporter intentionally has fixed runtime batch 1.
            actual = session.run(["actions"], {"obs": flat.numpy()})[0]
            np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-4, atol=2e-5)
    with pytest.raises(FileExistsError):
        export_policy(directory, task, path.parent, encoder=encoder)


def test_external_goal_indices_layout_and_actual_root(synthetic_bundles):
    directory, task, _ = synthetic_bundles["gencoord", "proprio"]
    bundle = load_policy_bundle(directory, task)
    q = torch.arange(80, dtype=torch.float32)[:, None].expand(80, 23).clone()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(80, 4)
    motion = SimpleNamespace(
        joint_pos=q,
        joint_vel=q + 100,
        root_pos_w=torch.zeros(80, 3),
        root_quat_w=quat,
        motion_ends=torch.tensor([12, 80]),
    )
    # Nonidentity actual root makes actual-relative vs reference-relative detectable.
    root_pose = torch.tensor([[0.0, 0.0, 0.5, 0.70710678, 0.0, 0.0, 0.70710678]])
    goal = motion_goal(bundle, motion, torch.tensor([1]), root_pose)
    indices = [1, 6, 11, 11, 11, 11, 11, 11, 11, 11]
    expected_content = torch.cat([torch.cat([q[t], q[t] + 100]) for t in indices])
    torch.testing.assert_close(goal[0, :460], expected_content)
    torch.testing.assert_close(
        goal[0, 460:].reshape(10, 6),
        torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]).expand(10, 6),
        atol=1e-6,
        rtol=0,
    )
    assert not torch.equal(
        goal[0, 460:466], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    )
    # Explicit small tagged example detects accidental temporal flattening.
    hist = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    torch.testing.assert_close(
        pack_input(hist, torch.tensor([[7.0]]), [1, 2]),
        torch.tensor([[1.0, 4.0, 2.0, 3.0, 5.0, 6.0, 7.0]]),
    )


@pytest.mark.parametrize(
    "failure",
    [
        "partial_cfg",
        "unknown_cfg",
        "wrong_history",
        "wrong_state",
        "negative_std",
        "wrong_scale",
    ],
)
def test_mismatched_bundle_is_rejected(tmp_path, synthetic_bundles, failure):
    directory, task, _ = synthetic_bundles["gencoord", "proprio"]
    for name in ("manifest.json", "actor.pt", "world_model.pt"):
        (tmp_path / name).write_bytes((directory / name).read_bytes())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    if failure == "partial_cfg":
        del manifest["runner_cfg"]["actor_hidden_dims"]
    elif failure == "unknown_cfg":
        manifest["runner_cfg"]["unexpected"] = True
    elif failure == "wrong_history":
        manifest["runner_cfg"]["history_length"] = 9
    elif failure in ("wrong_state", "negative_std"):
        state = torch.load(tmp_path / "actor.pt", weights_only=True)
        state["obs_normalizer._std"] = (
            torch.ones(1, 1) if failure == "wrong_state" else -torch.ones(1, 75)
        )
        torch.save(state, tmp_path / "actor.pt")
    else:
        torch.save(
            {
                "dynamics": {
                    "action_scale": torch.ones(1, 23),
                    "action_offset": torch.zeros(1, 23),
                }
            },
            tmp_path / "world_model.pt",
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, RuntimeError)):
        export_policy(tmp_path, task, tmp_path / "out")
    assert not (tmp_path / "out").exists()
