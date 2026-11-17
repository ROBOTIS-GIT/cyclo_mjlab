"""Motion-independent FP32 ONNX export of the original deployed Blink Policy."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .runtime import VENDOR, prepare


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class FlatPolicy(nn.Module):
    """Shape adapter only: temporal raw history followed by an external raw goal."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.hist = policy.hist
        self.obs_dim = policy.obs_norm.mean.numel()
        self.history_size = self.hist * self.obs_dim
        self.input_size = self.history_size + policy.goal_norm.mean.numel()

    def forward(self, observations):
        history = observations[:, : self.history_size].reshape(
            -1, self.hist, self.obs_dim
        )
        return self.policy(history, observations[:, self.history_size :])


def load_policy(checkpoint, robot):
    """Rebuild the recorded architecture strictly; never substitute live weights."""
    prepare()
    from blinkmimic import networks
    from blinkmimic.mdp import obs_dim
    from blinkmimic.networks import Policy, check_pruned, policy_kwargs

    if networks.AMP_DTYPE is not None:
        raise ValueError("FP32 export requires an isolated process with AMP_DTYPE=None")
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected an official Blink checkpoint dictionary")
    args, state = checkpoint.get("args"), checkpoint.get("pi")
    if not isinstance(args, dict) or not isinstance(state, dict):
        raise ValueError("Expected official Blink checkpoint with args and deployed pi")
    required = (
        "robot",
        "hist",
        "goal_frames",
        "enc_hidden",
        "dec_hidden",
        "pi_ref_skip",
    )
    if any(key not in args for key in required):
        raise ValueError("Checkpoint is missing required policy configuration")
    if args["robot"] != robot.name:
        raise ValueError("Checkpoint robot does not match the selected robot")
    for key in ("hist", "goal_frames"):
        if type(args[key]) is not int or args[key] < 1:
            raise ValueError(f"Invalid checkpoint {key}")
    if type(args["pi_ref_skip"]) is not bool:
        raise ValueError("Invalid checkpoint pi_ref_skip")
    check_pruned(args, nets_only=True)

    def widths(key):
        try:
            result = tuple(int(v) for v in args[key].split(","))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid checkpoint {key}") from error
        if not result or any(v < 1 for v in result):
            raise ValueError(f"Invalid checkpoint {key}")
        return result

    joints = robot.num_joints
    q0 = torch.tensor(robot.default_joint_pos_list(), dtype=torch.float32)
    scale = torch.tensor(robot.action_scale_list(), dtype=torch.float32)
    if (
        not torch.isfinite(q0).all()
        or not torch.isfinite(scale).all()
        or (scale == 0).any()
    ):
        raise ValueError("Invalid robot nominal pose/action scale")
    if args["pi_ref_skip"]:
        for name, expected in (("ref_q0", q0), ("ref_alpha", scale)):
            if name not in state or not torch.equal(state[name], expected):
                raise ValueError(f"Checkpoint {name} differs from the robot contract")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"Expected FP32 policy tensor: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite policy tensor: {name}")
        if name.endswith(".var") and (value < 0).any():
            raise ValueError(f"Negative normalizer variance: {name}")
    policy = Policy(
        obs_dim(joints),
        args["goal_frames"] * (15 + 2 * joints),
        joints,
        hist=args["hist"],
        enc_hidden=widths("enc_hidden"),
        dec_hidden=widths("dec_hidden"),
        **policy_kwargs(args, robot),
    )
    policy.load_state_dict(state, strict=True)
    return FlatPolicy(policy.eval()).eval()


def contract(wrapper, checkpoint, robot):
    """Machine-readable external feature and action contract, with no clip assets."""
    j = robot.num_joints

    def fields(items):
        result, offset = [], 0
        for name, width, formula in items:
            result.append(dict(name=name, offset=offset, width=width, formula=formula))
            offset += width
        return result

    return dict(
        schema="cyclo.blinkmimic.external_goal.v1",
        upstream=json.loads((VENDOR / "UPSTREAM.json").read_text())["commit"],
        weights_key="pi",
        pi_ema=checkpoint["args"].get("pi_ema", 0.0),
        source_amp=checkpoint["args"].get("amp", "none"),
        export_precision="float32",
        precision_scope="FP32 Policy equivalence; not bitwise BF16 CUDA equivalence",
        input=dict(
            name="observations",
            dtype="float32",
            shape=["batch", wrapper.input_size],
            layout="temporal history flattened oldest first, then flattened future goals",
            history_shape=[wrapper.hist, wrapper.obs_dim],
            goal_shape=[checkpoint["args"]["goal_frames"], 15 + 2 * j],
        ),
        output=dict(
            name="actions",
            dtype="float32",
            shape=["batch", j],
            meaning="final normalized action including reference skip if enabled",
        ),
        normalization=dict(
            formula="(x-mean)/sqrt(var+eps**2)",
            eps=0.01,
            scope="inside graph, frozen pi obs_norm and goal_norm; no clipping",
        ),
        observation_fields=fields(
            [
                ("joint_position_relative", j, "q-q0-joint_offset"),
                ("joint_velocity", j, "qdot, raw rad/s"),
                (
                    "root_angular_velocity_body",
                    3,
                    "R_actual.T @ omega_world, raw rad/s",
                ),
                ("projected_gravity_body", 3, "-R_actual[2,:]"),
                (
                    "previous_action",
                    j,
                    "previous final normalized action including skip",
                ),
                ("reference_relative_orientation_6d", 6, "sixd(R_actual.T @ R_ref[f])"),
            ]
        ),
        goal_fields_per_future_frame=fields(
            [
                ("root_displacement", 3, "H_ref[f].T @ (p_ref[t]-p_ref[f])"),
                ("root_orientation_6d", 6, "sixd(H_ref[f].T @ R_ref[t])"),
                ("root_linear_velocity_body", 3, "R_ref[t].T @ v_world_link_origin[t]"),
                ("root_angular_velocity_body", 3, "R_ref[t].T @ omega_world[t]"),
                ("joint_position", j, "q_ref[t]"),
                ("joint_velocity_scaled", j, "0.1*qdot_ref[t]"),
            ]
        ),
        future_frames="t=min(f+k, current_clip_last_frame), k=1..goal_frames",
        frame=dict(
            quaternion="wxyz",
            rotation="R maps root body column vectors to world",
            sixd="[R00,R01,R10,R11,R20,R21]",
            linear_velocity_point="root link origin",
            yaw="H from projected reference x axis; norm sqrt(R00**2+R10**2+1e-9)",
            quaternion_normalization="torch.nn.functional.normalize, eps=1e-12",
        ),
        history_reset="repeat initial observation hist times; previous action zero",
        reference_skip=dict(
            enabled=wrapper.policy.ref_off is not None,
            raw_goal_offset=wrapper.policy.ref_off,
            formula="(q_ref[f+1]-q0)/action_scale + decoder_residual",
        ),
        action=dict(
            joint_order=list(robot.joint_names),
            q0=robot.default_joint_pos_list(),
            scale=robot.action_scale_list(),
            target="q0 + action_scale * actions (nominal joint_offset=0)",
            clipping="none; do not add q_ref again",
        ),
        external_state="history, reference frame alignment, goal features, clip boundaries",
        includes_motion=False,
    )


def export_policy(checkpoint_path, output_dir, robot):
    """Export on CPU without a simulator, dataset, selector or robot connection."""
    import onnx

    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    wrapper = load_policy(checkpoint, robot)
    metadata = contract(wrapper, checkpoint, robot)
    metadata["checkpoint_sha256"] = sha256(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (torch.zeros(1, wrapper.input_size),),
            str(path),
            export_params=True,
            opset_version=18,
            input_names=["observations"],
            output_names=["actions"],
            dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
            dynamo=False,
        )
    model = onnx.load(path)
    onnx.helper.set_model_props(model, {"cyclo.policy_contract": json.dumps(metadata)})
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    metadata["onnx_sha256"] = sha256(path)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare()
    from .robot import make_robot

    print(export_policy(args.checkpoint, args.output_dir, make_robot()))


if __name__ == "__main__":
    main()
