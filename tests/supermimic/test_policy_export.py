"""
Synthetic full-width encoder export; no trained weights or motion datasets.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.entity import Entity
from supermimic.representations import create_representation
from supermimic.runners.supermimic_runner import (
    SuperMimicRunnerCfg,
    build_actor,
)
from supermimic.tasks.task_spec import build_actor_obs_td

from source.utils.export_notices import distribution_license
from source.tasks.supermimic import REPO_ROOT
from source.tasks.supermimic.checkpoint_config import make_manifest
from source.tasks.supermimic.export import (
    export_policy,
    load_policy_bundle,
    motion_goal,
    nominal_action_contract,
    pack_input,
    sha256,
)
from source.tasks.supermimic.motion import model_joint_names
from source.tasks.supermimic.config import TASKS, register_tasks
from source.tasks.supermimic.config.k1_rev1.base_env_cfg import make_env_cfg


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
        task = (
            "Cyclo-SuperMimic-"
            f"{'GenCoord' if representation == 'gencoord' else 'MaxCoord'}"
            "-K1-Rev1-001"
        )
        cfg = make_env_cfg(representation)
        cfg.commands["motion"].sampling_mode = "external"
        rcfg = SuperMimicRunnerCfg(
            representation=representation,
            actor_obs=mode,
            history_length=history,
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
            for norm in [
                actor.obs_normalizer,
                *actor.goal_normalizers.values(),
            ]:
                norm._mean.copy_(
                    torch.linspace(-0.2, 0.3, norm._mean.numel())[None]
                )
                norm._std.copy_(
                    torch.linspace(0.4, 1.4, norm._std.numel())[None]
                )
                # Deliberately not std**2: export must preserve saved std, not
                # recompute.
                norm._var.fill_(5)
                norm.count.fill_(12345)
        directory = tmp_path_factory.mktemp(
            f"synthetic-{representation}-{mode}"
        )
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

    wrapper = _SingleEncoderOnnx(
        bundle.actor, encoder, bundle.term_widths
    ).eval()
    metadata = json.loads((path.parent / "metadata.json").read_text())
    expected_width = (
        1270
        if mode == "proprio"
        else (576 if representation == "gencoord" else 734)
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
    assert metadata["source_sha256"]["actor.pt"] == sha256(
        directory / "actor.pt"
    )
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    assert [i.name for i in model.graph.input] == ["obs"]
    assert [o.name for o in model.graph.output] == ["actions"]
    embedded = json.loads(model.metadata_props[0].value)
    assert embedded == {
        k: v for k, v in metadata.items() if k != "onnx_sha256"
    }
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
        if (
            batch == 1
        ):  # Upstream exporter intentionally has fixed runtime batch 1.
            actual = session.run(["actions"], {"obs": flat.numpy()})[0]
            np.testing.assert_allclose(
                actual, expected.numpy(), rtol=2e-4, atol=2e-5
            )
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
    # Nonidentity actual root makes actual-relative vs reference-relative
    # detectable.
    root_pose = torch.tensor(
        [[0.0, 0.0, 0.5, 0.70710678, 0.0, 0.0, 0.70710678]]
    )
    goal = motion_goal(bundle, motion, torch.tensor([1]), root_pose)
    indices = [1, 6, 11, 11, 11, 11, 11, 11, 11, 11]
    expected_content = torch.cat(
        [torch.cat([q[t], q[t] + 100]) for t in indices]
    )
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
            torch.ones(1, 1)
            if failure == "wrong_state"
            else -torch.ones(1, 75)
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


@pytest.mark.parametrize(
    "representation,mode,active_encoders",
    [
        ("gencoord", "proprio", ("motion",)),
        ("maxcoord", "proprio", ("motion", "velocity")),
        ("gencoord", "rep_state", ("motion", "velocity")),
        ("maxcoord", "rep_state", ("velocity",)),
    ],
)
def test_training_start_writes_deployment_configs(
    synthetic_bundles,
    tmp_path,
    monkeypatch,
    representation,
    mode,
    active_encoders,
):
    import yaml
    from source.tasks.supermimic.checkpoint import LocalBundleRunner
    from source.tasks.supermimic.export import policy_contract

    directory, task, _ = synthetic_bundles[representation, mode]
    bundle = load_policy_bundle(directory, task)
    runner = object.__new__(LocalBundleRunner)
    runner.manifest = bundle.manifest
    runner._active_task_specs = [
        spec for spec in bundle.actor.task_specs if spec.name in active_encoders
    ]
    runner.cfg = bundle.runner_cfg
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=bundle.env_cfg))
    runner.policy_learner = SimpleNamespace(
        actor=bundle.actor, actor_optimizers=[]
    )
    runner.world_model_learner = SimpleNamespace(optimizers=[])
    runner._checkpoint_step = 0
    runner._wm = SimpleNamespace(
        bundle=lambda: torch.load(
            directory / "world_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    runner._representation = bundle.representation
    runner.logger = SimpleNamespace()
    monkeypatch.chdir(tmp_path)
    runner.task_id = task

    def unexpected_metadata(*args, **kwargs):
        pytest.fail("Training YAML export must not construct ONNX metadata")

    monkeypatch.setattr(
        "source.tasks.supermimic.export.policy_contract", unexpected_metadata
    )

    run_dir = runner._create_log_dir()
    assert run_dir.parent == Path("logs/supermimic") / task
    assert not (run_dir / "actor.pt").exists()
    assert not (run_dir / "world_model.pt").exists()
    expected_files = [str(run_dir / "manifest.json")]
    for encoder, relative in (
        ("motion", "params/sim2real.yaml"),
        ("velocity", "params/velocity/sim2real.yaml"),
    ):
        path = run_dir / relative
        if encoder not in active_encoders:
            assert not path.exists()
            continue
        expected_files.append(str(path))
        cfg = yaml.safe_load(path.read_text())
        contract = policy_contract(bundle, encoder)
        obs = cfg["observations"]["supermimic"]
        assert cfg["step_dt"] == 0.02
        assert cfg["policy_joints"] == contract["action"]["joint_order"]
        assert cfg["actions"]["joint_pos"]["scale"] == [
            float(f"{value:.3g}") for value in contract["action"]["scale"]
        ]
        assert cfg["actions"]["joint_pos"]["offset"] == [
            float(f"{value:.3g}") for value in contract["action"]["offset"]
        ]
        assert "input" not in obs
        assert "body_order" not in obs
        assert "history_reset" not in obs
        assert "normalization_in_onnx" not in obs
        assert "imu_site_quaternion_wxyz" not in obs
        assert obs.get(
            "default_joint_velocity", [0.0] * len(cfg["policy_joints"])
        ) == [float(f"{value:.3g}") for value in contract["action"]["qdot0"]]
        assert all(
            term["history_length"] == bundle.actor.history_length
            for term in obs["history"]["terms"].values()
        )
        assert obs["actor_obs"] == mode
        if encoder == "motion":
            assert (
                obs["motion_goal"]
                == contract["observation_config"]["motion_goal"]
            )
            assert [
                list(group["terms"]) for group in obs["motion_goal"]["groups"]
            ] == [
                ["future_joint_pos", "future_joint_vel"],
                ["future_root_orientation"],
            ]
            assert not cfg["commands"]
        else:
            assert "motion_goal" not in obs
            assert cfg["commands"]["base_velocity"]["ranges"][
                "lin_vel_x"
            ] == list(bundle.runner_cfg.tasks.velocity.lin_vel_x_range)
        model = Entity(bundle.env_cfg.scene.entities["robot"]).spec.compile()
        for name, props in cfg["joint_properties"].items():
            aid = model.actuator(name).id
            assert props["stiffness"] == float(
                f"{model.actuator_gainprm[aid, 0]:.3g}"
            )
            assert props["damping"] == float(
                f"{-model.actuator_biasprm[aid, 2]:.3g}"
            )
    assert runner.checkpoint_extra_files(run_dir) == expected_files
    runner.logger.close()


def test_observation_preprocessing_uses_resolved_training_settings(
    synthetic_bundles, tmp_path
):
    import yaml
    from source.tasks.supermimic.export import policy_contract
    from source.tasks.supermimic.export_sim2real_cfg import export_sim2real_cfg

    directory, task, _ = synthetic_bundles["gencoord", "proprio"]
    bundle = load_policy_bundle(directory, task)
    source = bundle.env_cfg.observations["actor"].terms["base_ang_vel"]
    source.scale = (0.123456, 0.25, 0.5)
    source.clip = (-12.3456, 12.3456)

    contract = policy_contract(bundle, "motion")
    term = contract["observation_config"]["history"]["terms"]["base_ang_vel"]
    assert term == dict(
        scale=list(source.scale), clip=source.clip, history_length=10
    )

    path = export_sim2real_cfg(
        tmp_path,
        bundle.env_cfg,
        bundle.runner_cfg,
        bundle.representation,
        bundle.action["joint_order"],
    )
    observation = yaml.safe_load(path.read_text())["observations"][
        "supermimic"
    ]
    assert observation["history"]["terms"]["base_ang_vel"] == dict(
        scale=[0.123, 0.25, 0.5],
        clip=[-12.3, 12.3],
        history_length=10,
    )
    assert (
        observation["motion_goal"]
        == contract["observation_config"]["motion_goal"]
    )


def test_observation_order_must_match_native_representation(synthetic_bundles):
    from source.tasks.supermimic.export import policy_contract

    directory, task, _ = synthetic_bundles["gencoord", "proprio"]
    bundle = load_policy_bundle(directory, task)
    group = bundle.env_cfg.observations["actor"]
    group.terms = dict(reversed(list(group.terms.items())))
    with pytest.raises(ValueError, match="observation order"):
        policy_contract(bundle, "motion")


def test_play_exports_before_simulator_and_refreshes_existing_files(
    synthetic_bundles,
    tmp_path,
    monkeypatch,
):
    import shutil
    from source.tasks.supermimic import play

    source, task, _ = synthetic_bundles["gencoord", "proprio"]
    directory = tmp_path / "checkpoint"
    shutil.copytree(source, directory)
    params = directory / "params" / "sim2real.yaml"
    params.parent.mkdir()
    params.write_text("training-owned yaml\n")

    class BeforeSimulator(Exception):
        pass

    def stop_before_simulator(_):
        raise BeforeSimulator

    monkeypatch.setattr(play, "setup_device", stop_before_simulator)
    cfg = play.PlayConfig(motion_file="unused.npz", checkpoint_dir=directory)
    assert cfg.viewer == "viser"
    for iteration in range(2):
        with pytest.raises(BeforeSimulator):
            play.play(task, cfg)
        for relative in (
            "exported/policy.onnx",
            "exported/velocity/policy.onnx",
        ):
            path = directory / relative
            metadata = json.loads((path.parent / "metadata.json").read_text())
            assert metadata["onnx_sha256"] == sha256(path)
            assert metadata["source_sha256"]["actor.pt"] == sha256(
                directory / "actor.pt"
            )
        assert params.read_text() == "training-owned yaml\n"
        path = directory / "exported/policy.onnx"
        if iteration == 0:
            previous_hash = sha256(path)
            state = torch.load(directory / "actor.pt", weights_only=True)
            key = next(
                key
                for key in state
                if key.startswith("decoder") and key.endswith("bias")
            )
            state[key] += 0.1
            torch.save(state, directory / "actor.pt")
        else:
            assert sha256(path) != previous_hash
    assert sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("sim2real.yaml")
    ) == ["params/sim2real.yaml"]


def test_play_can_skip_export(synthetic_bundles, monkeypatch):
    from source.tasks.supermimic import play, export

    directory, task, _ = synthetic_bundles["gencoord", "proprio"]
    monkeypatch.setattr(
        export,
        "export_play_policies",
        lambda *args: pytest.fail("export called"),
    )

    class BeforeSimulator(Exception):
        pass

    def stop_before_simulator(_):
        raise BeforeSimulator

    monkeypatch.setattr(play, "setup_device", stop_before_simulator)
    with pytest.raises(BeforeSimulator):
        play.play(
            task,
            play.PlayConfig(
                motion_file="unused.npz",
                checkpoint_dir=directory,
                export_policy=False,
            ),
        )


@pytest.mark.parametrize("representation", ["gencoord", "maxcoord"])
def test_exported_history_matches_native_observation_functions(
    synthetic_bundles, representation
):
    from unittest.mock import Mock
    from mjlab.sensor import BuiltinSensor
    from source.tasks.supermimic.export import policy_contract
    from supermimic.representations import term_major_to_temporal

    directory, task, _ = synthetic_bundles[representation, "proprio"]
    bundle = load_policy_bundle(directory, task)
    torch.manual_seed(121)
    h = bundle.runner_cfg.history_length
    n = h * 2
    q, qd, bias = (torch.randn(n, 23) for _ in range(3))
    q0, qd0 = torch.randn(1, 23), torch.randn(1, 23)
    gravity, gyro, action = (
        torch.randn(n, 3),
        torch.randn(n, 3),
        torch.randn(n, 23),
    )
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    projected_gravity_b=gravity,
                    joint_pos=q,
                    joint_pos_biased=q + bias,
                    default_joint_pos=q0,
                    joint_vel=qd,
                    default_joint_vel=qd0,
                )
            ),
            "robot/imu_ang_vel": Mock(spec=BuiltinSensor, data=gyro),
        },
        action_manager=SimpleNamespace(action=action),
    )
    # Exercise the native clip-then-scale pipeline as well as distinct joint
    # biases.
    bundle.env_cfg.observations["actor"].terms["base_ang_vel"].clip = (
        -0.4,
        0.4,
    )
    bundle.env_cfg.observations["actor"].terms["base_ang_vel"].scale = (
        0.1,
        0.2,
        0.3,
    )
    native = []
    for term in bundle.env_cfg.observations["actor"].terms.values():
        value = term.func(env, **term.params)
        if term.clip is not None:
            value = value.clamp(*term.clip)
        if term.scale is not None:
            value = value * torch.tensor(term.scale)
        native.append(value)
    expected = torch.cat(native, dim=-1).reshape(2, h, -1)

    group = policy_contract(bundle, "motion")["observation_config"]["history"]
    values = dict(
        projected_gravity=gravity,
        base_ang_vel=gyro,
        joint_pos_rel=q + bias - q0,
        joint_vel_rel=qd - qd0,
        last_action=action,
    )
    packed = []
    for name, cfg in group["terms"].items():
        value = values[name]
        if cfg["clip"] is not None:
            value = value.clamp(*cfg["clip"])
        value = value * torch.tensor(cfg["scale"])
        packed.append(value.reshape(2, h, -1).flatten(1))
    actual = term_major_to_temporal(
        torch.cat(packed, -1), bundle.term_widths, h
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("representation", ["gencoord", "maxcoord"])
def test_exported_goal_groups_match_native_extractor(
    synthetic_bundles, representation
):
    from mjlab.utils.lab_api.math import matrix_from_quat
    from source.tasks.supermimic.export import policy_contract

    directory, task, _ = synthetic_bundles[representation, "proprio"]
    bundle = load_policy_bundle(directory, task)
    torch.manual_seed(492)
    quaternions = torch.nn.functional.normalize(torch.randn(80, 4), dim=-1)
    motion = SimpleNamespace(
        joint_pos=torch.randn(80, 23),
        joint_vel=torch.randn(80, 23) * 4,
        root_pos_w=torch.randn(80, 3),
        root_quat_w=quaternions,
        motion_ends=torch.tensor([12, 80]),
    )
    # The physical cursor is one frame before native command.time_steps.
    current = torch.tensor([0, 9, 10, 12, 77, 78])
    root_pose = torch.cat(
        [
            torch.randn(6, 3),
            torch.nn.functional.normalize(torch.randn(6, 4), dim=-1),
        ],
        -1,
    )
    cfg = policy_contract(bundle, "motion")["observation_config"][
        "motion_goal"
    ]
    selected = (
        current[:, None]
        + cfg["start_offset"]
        + torch.arange(cfg["frames"]) * cfg["stride"]
    )
    end = torch.where(current < 12, 11, 79)
    selected = selected.minimum(end[:, None])
    relative = matrix_from_quat(root_pose[:, 3:]).transpose(-1, -2)[
        :, None
    ] @ matrix_from_quat(quaternions[selected])
    values = dict(
        future_joint_pos=motion.joint_pos[selected],
        future_joint_vel=motion.joint_vel[selected],
        future_root_orientation=relative[..., :2].reshape(6, cfg["frames"], 6),
    )
    groups = []
    for group in cfg["groups"]:
        parts = [
            values[name] * torch.tensor(term["scale"])
            for name, term in group["terms"].items()
        ]
        groups.append(torch.cat(parts, -1).flatten(1))
    actual = torch.cat(groups, -1)
    expected = motion_goal(bundle, motion, current + 1, root_pose)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_export_rejects_changed_observation_calculation(synthetic_bundles):
    from source.tasks.supermimic.export import policy_contract

    directory, task, _ = synthetic_bundles["gencoord", "proprio"]
    bundle = load_policy_bundle(directory, task)
    bundle.env_cfg.observations["actor"].terms["joint_pos"].params[
        "biased"
    ] = False
    with pytest.raises(ValueError, match="calculation differs"):
        policy_contract(bundle, "motion")


def test_historical_archive_exports_saved_actor_without_split_files(
    synthetic_bundles, tmp_path
):
    import onnxruntime as ort
    from source.tasks.supermimic.checkpoint_archive import (
        save_checkpoint,
        load_checkpoint,
    )
    from source.tasks.supermimic.export import export_play_policies

    directory, task, _ = synthetic_bundles["gencoord", "proprio"]
    latest = load_checkpoint(directory)
    path = save_checkpoint(
        tmp_path,
        step=1000,
        manifest=latest.manifest,
        actor=latest.actor,
        world_model=latest.world_model,
    )
    # The archive remains self-contained even after being moved away from the
    # run.
    moved = tmp_path / "moved" / path.name
    moved.parent.mkdir()
    path.rename(moved)
    expected = export_policy(directory, task, tmp_path / "expected")
    outputs = export_play_policies(moved, task)
    assert outputs[0] == tmp_path / "moved/exported/model_1000/policy.onnx"
    assert not (tmp_path / "actor.pt").exists()
    a = ort.InferenceSession(str(expected), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(
        str(outputs[0]), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(42)
    obs = rng.normal(size=a.get_inputs()[0].shape).astype(np.float32)
    np.testing.assert_array_equal(
        a.run(None, {"obs": obs})[0], b.run(None, {"obs": obs})[0]
    )
    metadata = json.loads((outputs[0].parent / "metadata.json").read_text())
    assert metadata["checkpoint_step"] == 1000
    assert metadata["source_sha256"] == {moved.name: sha256(moved)}


def test_play_export_loads_one_actor_for_all_encoders(
    synthetic_bundles, monkeypatch
):
    from unittest.mock import Mock
    from source.tasks.supermimic import export

    directory, task, _ = synthetic_bundles["gencoord", "proprio"]
    loader = Mock(wraps=export.load_policy_bundle)
    monkeypatch.setattr(export, "load_policy_bundle", loader)
    bundles = []

    def export_encoder(checkpoint, bundle, output, *, encoder, overwrite):
        bundles.append((bundle, encoder))
        return output / "policy.onnx"

    monkeypatch.setattr(export, "export_encoder", export_encoder)
    paths = export.export_play_policies(directory, task)
    assert loader.call_count == 1
    assert [encoder for _, encoder in bundles] == ["motion", "velocity"]
    assert bundles[0][0] is bundles[1][0]
    assert paths == [
        directory / "exported/policy.onnx",
        directory / "exported/velocity/policy.onnx",
    ]
