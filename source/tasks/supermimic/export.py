"""
Motion-independent SuperMimic policy export using the shared encoder slice.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from supermimic.models import MultiEncoderAE
from supermimic.representations import (
    GOAL_FRAME_STRIDE,
    NUM_GOAL_FRAMES,
    StateRepresentation,
    create_representation,
)
from supermimic.runners.exporter import _export_module, _SingleEncoderOnnx
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.tasks.task_spec import resolve_task_specs

from source.utils.export_notices import (
    distribution_license,
    write_software_notices,
)

from . import REPO_ROOT
from .checkpoint_config import json_value, read_bundle
from .checkpoint_archive import load_checkpoint
from .config import TASKS, register_tasks
from .export_sim2real_cfg import (
    observation_config,
    observation_terms,
    resolve_action_settings,
    validate_policy_config,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class PolicyBundle:
    actor: MultiEncoderAE
    representation: StateRepresentation
    runner_cfg: SuperMimicRunnerCfg
    env_cfg: ManagerBasedRlEnvCfg
    manifest: dict
    action: dict
    term_widths: list[int]


def nominal_action_contract(cfg, joint_names):
    """Add model-frame documentation to the deployment action settings."""
    from mjlab.entity import Entity

    action = resolve_action_settings(cfg, joint_names)
    model = Entity(cfg.scene.entities["robot"]).spec.compile()
    imu = model.site("imu")
    return dict(
        **action,
        imu_site_body=model.body(int(model.site_bodyid[imu.id])).name,
        imu_site_quaternion_wxyz=model.site_quat[imu.id].tolist(),
    )


def validate_saved_action(wm, action):
    # These are the actual affine buffers used by the trained dynamics model.
    if not isinstance(wm, dict) or not isinstance(wm.get("dynamics"), dict):
        raise ValueError(
            "Expected a native world_model.pt dynamics dictionary"
        )
    for key, expected in (
        ("action_scale", action["scale"]),
        ("action_offset", action["offset"]),
    ):
        saved = wm.get("dynamics", {}).get(key)
        if (
            not isinstance(saved, torch.Tensor)
            or saved.numel() != len(action["scale"])
            or not torch.equal(
                saved.reshape(-1), torch.tensor(expected, dtype=torch.float32)
            )
        ):
            raise ValueError(
                f"Saved dynamics {key} differs from the native action contract"
            )


def validate_actor_state(state):
    if not isinstance(state, dict):
        raise ValueError("Expected an actor.pt state dictionary")
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid policy tensor: {name}")
        if name.endswith("count"):
            if value.dtype != torch.int64 or (value < 0).any():
                raise ValueError(f"Invalid normalizer count: {name}")
        elif value.dtype != torch.float32:
            raise ValueError(f"Expected FP32 policy tensor: {name}")
        if name.endswith(("._var", "._std")) and (value < 0).any():
            raise ValueError(f"Negative normalizer buffer: {name}")


def load_policy_bundle(directory, task):
    """
    Load saved settings, actor and affine buffers; never convert observation
    mode.
    """
    from mjlab.tasks.registry import list_tasks

    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    checkpoint = load_checkpoint(directory)
    manifest = checkpoint.manifest
    cfg, rcfg = read_bundle(checkpoint, task)
    # The generic loader supports old partial configs; export must not guess
    # them.
    if json_value(rcfg) != manifest.get("runner_cfg"):
        raise ValueError(
            "Export requires a complete, recognized saved runner_cfg"
        )
    validate_policy_config(cfg, rcfg, task)
    names = manifest["robot"]["joint_order"]
    action = nominal_action_contract(cfg, names)
    validate_saved_action(checkpoint.world_model, action)
    j = len(names)
    bodies = len(cfg.commands["motion"].body_names)
    rep = create_representation(
        rcfg.representation,
        13 + 2 * j if rcfg.representation == "gencoord" else 13 * bodies,
        j,
        cfg.sim.mujoco.timestep * cfg.decimation,
        cfg.decimation,
        torch.tensor(action["q0"], dtype=torch.float32),
        actor_obs_mode=rcfg.actor_obs,
    )
    actor = MultiEncoderAE(
        obs_dim=(
            rep.rep_state_dim
            if rcfg.actor_obs == "rep_state"
            else rep.proprio_obs_dim(j)
        ),
        action_dim=j,
        task_specs=resolve_task_specs(
            rcfg.tasks, rep, NUM_GOAL_FRAMES * (2 * j + 9), "cpu"
        ),
        slot_mix=rcfg.slot_mix,
        latent_dim=rcfg.latent_dim,
        hidden_dims=rcfg.actor_hidden_dims,
        latent_bc_weight=rcfg.latent_bc_weight,
        history_length=rcfg.history_length,
    )
    state = checkpoint.actor
    validate_actor_state(state)
    actor.load_state_dict(state, strict=True)
    widths = (
        rep.proprio_term_widths(j)
        if rcfg.actor_obs == "proprio"
        else [rep.rep_state_dim]
    )
    return PolicyBundle(actor.eval(), rep, rcfg, cfg, manifest, action, widths)


def pack_input(history, goal, term_widths):
    """
    External temporal [B,H,D] history -> upstream term-major flat input + goal.
    """
    if (
        history.ndim != 3
        or goal.ndim != 2
        or history.shape[0] != goal.shape[0]
    ):
        raise ValueError("Expected history[B,H,D] and goal[B,G]")
    if sum(term_widths) != history.shape[-1] or any(
        w < 1 for w in term_widths
    ):
        raise ValueError("Observation term widths do not match history")
    return torch.cat(
        [part.flatten(1) for part in history.split(term_widths, dim=-1)]
        + [goal],
        dim=-1,
    )


def motion_goal(bundle, motion, frames, root_pose_w):
    """
    Use the original training goal extractor with external arrays and root
    pose.

    motion supplies joint_pos, joint_vel, root_pos_w, root_quat_w and
    motion_ends
    (exclusive clip ends), exactly as the native MotionLoader does. root_pose_w
    is the CURRENT ACTUAL root [position(3), quaternion wxyz(4)], not the
    reference.
    """
    spec = next(ts for ts in bundle.actor.task_specs if ts.name == "motion")
    return spec.extract_training_goal(motion, frames, root_pose_w)


def policy_contract(bundle, encoder):
    actor, rcfg = bundle.actor, bundle.runner_cfg
    spec = next((ts for ts in actor.task_specs if ts.name == encoder), None)
    if spec is None:
        raise ValueError(f"Unknown policy encoder: {encoder}")
    j = len(bundle.action["joint_order"])
    h = rcfg.history_length
    terms = observation_terms(bundle.env_cfg, rcfg, bundle.representation, j)
    if rcfg.actor_obs == "proprio":
        native_formulas = {
            "projected_gravity": "R_root.T @ [0,0,-1]",
            "base_ang_vel": "pelvis IMU gyro in IMU site axes, rad/s",
            "joint_pos": "q + encoder_bias - q0, rad",
            "joint_vel": "qdot - qdot0, rad/s",
            "actions": "previous raw policy output, no reference addition",
        }
        formulas = [native_formulas[term["name"]] for term in terms]
    else:
        formulas = [
            (
                "[root_z(1), gravity_body(3), root_v_body(3),"
                " root_omega_body(3), absolute_q(J), qdot(J)]"
                if rcfg.representation == "gencoord"
                else (
                    "[root_relative_body_pos(3B),"
                    " root_relative_body_rot6d(6B),"
                    " body_linear_vel_in_root(3B),"
                    " body_angular_vel_in_root(3B), root_z(1),"
                    " gravity_body(3)]"
                )
            )
        ]
    for term, formula in zip(terms, formulas, strict=True):
        term["formula"] = formula
    if sum(term["width"] for term in terms) != actor.obs_dim:
        raise ValueError(
            "Observation term dimensions differ from the trained actor input"
        )
    offset = 0
    for term in terms:
        term["flat_offset"] = offset
        offset += term["width"] * h
    contract = dict(
        schema="cyclo.supermimic.external_goal.v1",
        algorithm="supermimic",
        task=bundle.manifest["task_id"],
        encoder=encoder,
        representation=rcfg.representation,
        actor_obs=rcfg.actor_obs,
        history_length=h,
        input=dict(
            name="obs",
            dtype="float32",
            shape=[1, h * actor.obs_dim + spec.goal_dim],
            layout=(
                "term-major history oldest first within each term, then"
                " external raw goal"
            ),
            observation_terms=terms,
            goal_offset=offset,
            goal_dim=spec.goal_dim,
        ),
        output=dict(
            name="actions",
            dtype="float32",
            shape=[1, j],
            meaning="raw decoder output; no clamp or reference-action skip",
        ),
        normalization=dict(
            formula="(x-saved_mean)/(saved_std+eps)",
            observation_eps=actor.obs_normalizer.eps,
            goal_eps=actor.goal_normalizers[encoder].eps,
            scope=(
                "inside graph, frozen saved _mean/_std, no variance"
                " recomputation or clipping"
            ),
        ),
        latent=dict(
            width=rcfg.latent_dim,
            formula="tanh(selected_encoder(normalized_goal))",
        ),
        action=dict(
            **bundle.action,
            formula="target=offset+scale*actions-encoder_bias",
            nominal_formula="target=offset+scale*actions",
            clipping="none",
        ),
        control_dt=bundle.env_cfg.sim.mujoco.timestep
        * bundle.env_cfg.decimation,
        body_order=list(bundle.env_cfg.commands["motion"].body_names),
        robot=bundle.manifest["robot"],
        motion_goal=(
            dict(
                frames=NUM_GOAL_FRAMES,
                stride=GOAL_FRAME_STRIDE,
                indices=(
                    "min(t+k*stride, current_clip_exclusive_end-1),"
                    " k=0..frames-1; t is native command.time_steps (reset at"
                    " physical frame f seeds t=f+1)"
                ),
                layout=(
                    f"all {NUM_GOAL_FRAMES} [absolute_joint_pos({j}),"
                    f" raw_joint_vel({j})] blocks, then all {NUM_GOAL_FRAMES}"
                    " relative_orientation_6d blocks"
                ),
                content_offset=0,
                orientation_offset=NUM_GOAL_FRAMES * 2 * j,
                orientation=(
                    "sixd(R_actual_current_root.T @ R_reference_future_root)"
                ),
                sixd="[R00,R01,R10,R11,R20,R21]",
                quaternion="wxyz",
                translation=(
                    "excluded from actor goal; present only in world-model"
                    " goal"
                ),
                scale=(
                    "no qdot/gyro prescale; no relative q0 subtraction in"
                    " motion goal"
                ),
            )
            if encoder == "motion"
            else dict(
                layout="[vx_body, vy_body, wz_body]",
                ranges=[
                    list(rcfg.tasks.velocity.lin_vel_x_range),
                    list(rcfg.tasks.velocity.lin_vel_y_range),
                    list(rcfg.tasks.velocity.ang_vel_z_range),
                ],
            )
        ),
        history_reset=(
            "zero previous action, repeat current observation H times; shift"
            " oldest out each control step"
        ),
        trained_slot_fraction=rcfg.slot_mix.get(encoder, 0.0),
        architecture=dict(
            hidden_dims=list(rcfg.actor_hidden_dims),
            latent_dim=rcfg.latent_dim,
        ),
        weights="actor.pt, no EMA substitution",
        precision="float32",
        includes_motion=False,
        includes_world_model=False,
        physical_robot_validation="not performed by export",
    )
    contract["observation_config"] = observation_config(
        bundle.env_cfg,
        rcfg,
        bundle.representation,
        j,
        encoder,
    )
    return contract


def export_policy(
    directory, task, output_dir, *, encoder="motion", overwrite=False
):
    checkpoint = load_checkpoint(directory)
    bundle = load_policy_bundle(checkpoint, task)
    return export_encoder(
        checkpoint, bundle, output_dir, encoder=encoder, overwrite=overwrite
    )


def export_encoder(
    checkpoint, bundle, output_dir, *, encoder, overwrite=False
):
    """Export one encoder from an already loaded and validated actor."""
    import onnx

    output_dir = Path(output_dir).expanduser()
    metadata = policy_contract(bundle, encoder)
    wrapper = _SingleEncoderOnnx(
        bundle.actor, encoder, bundle.term_widths
    ).eval()
    source = checkpoint.source
    metadata["source_sha256"] = (
        {
            name: sha256(source / name)
            for name in ("manifest.json", "actor.pt", "world_model.pt")
        }
        if source.is_dir()
        else {source.name: sha256(source)}
    )
    if checkpoint.step is not None:
        metadata["checkpoint_step"] = checkpoint.step
    metadata["shared_core_commit"] = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO_ROOT / "third_party/supermimic"),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    metadata["shared_exporter_sha256"] = sha256(
        REPO_ROOT / "third_party/supermimic/src/supermimic/runners/exporter.py"
    )
    output_dir.mkdir(parents=True, exist_ok=overwrite)
    metadata["software_notices"] = write_software_notices(
        output_dir,
        "Cyclo MJLab policy-only ONNX export\nCyclo adapter: Copyright 2026"
        " ROBOTIS CO., LTD."
        " (Apache-2.0).\n"
        "https://github.com/ROBOTIS-GIT/cyclo_mjlab\nSuperMimic"
        " policy/encoder slice: Copyright 2026 The SuperMimic Authors"
        " (Apache-2.0).\nOfficial source:"
        " https://github.com/HybridRobotics/supermimic\nShared source:"
        " https://github.com/ROBOTIS-GIT/supermimic\nThe graph retains the"
        " selected encoder, frozen normalizers and decoder; motion tables are"
        " omitted.\nRSL-RL EmpiricalNormalization (BSD-3-Clause):\nCopyright"
        " (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION\nCopyright (c) 2020"
        " Preferred Networks, Inc.\nKeep this notice and licenses/ with the"
        " exported graph when redistributing the bundle.\nThese software"
        " notices do not establish redistribution rights for learned weights"
        " or training data.\n",
        {
            "Apache-2.0.txt": REPO_ROOT / "third_party/supermimic/LICENSE",
            "RSL-RL-BSD-3-Clause.txt": distribution_license("rsl-rl-lib"),
        },
    )
    path = _export_module(
        wrapper,
        (torch.zeros(1, wrapper.input_size),),
        ["obs"],
        ["actions"],
        str(output_dir),
        "policy.onnx",
    )
    model = onnx.load(path)
    onnx.helper.set_model_props(
        model, {"cyclo.policy_contract": json.dumps(metadata)}
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    metadata["onnx_sha256"] = sha256(path)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return Path(path)


def export_play_policies(directory, task):
    """Refresh deployment graphs on play; training owns the YAML files."""
    checkpoint = load_checkpoint(directory)
    bundle = load_policy_bundle(checkpoint, task)
    paths = []
    for spec in bundle.actor.task_specs:
        encoder = spec.name
        output_dir = checkpoint.export_directory
        if encoder != "motion":
            output_dir /= encoder
        path = export_encoder(
            checkpoint, bundle, output_dir, encoder=encoder, overwrite=True
        )
        print(f"[INFO] Exported {encoder} policy: {path}")
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint-dir", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--encoder", choices=("motion", "velocity"), default="motion"
    )
    args = parser.parse_args(argv)
    selected = (
        args.checkpoint if args.checkpoint is not None else args.checkpoint_dir
    )
    if selected is None:
        from .play import latest_checkpoint_directory

        selected = latest_checkpoint_directory(Path("logs"), args.task)
    checkpoint = load_checkpoint(selected)
    output = args.output_dir or checkpoint.export_directory
    if args.output_dir is None and args.encoder != "motion":
        output /= args.encoder
    print(
        export_policy(
            checkpoint,
            args.task,
            output,
            encoder=args.encoder,
        )
    )


if __name__ == "__main__":
    main()
