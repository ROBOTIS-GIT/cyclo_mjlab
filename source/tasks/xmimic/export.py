"""Export the averaged X-Mimic policy with the shared external-goal contract."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from source.tasks.blinkmimic.export import (
    contract,
    load_policy,
    sha256,
    write_policy_bundle,
)
from source.tasks.blinkmimic.export_sim2real_cfg import export_sim2real_cfg
from source.tasks.blinkmimic.runtime import ROOT, prepare

from .config import TASKS, robot_contract
from .train import UPSTREAM


def read_checkpoint(path, robot):
    """Map X-Mimic's input-major dense kernels to the shared policy module."""
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["meta"].item())
        args = meta["args"]
        if args["robot"] != robot_contract(robot):
            raise ValueError("Checkpoint robot differs from the selected task")
        state = {}
        for name, option in (("enc", "enc_hidden"), ("dec", "dec_hidden")):
            layers = len(args[option].split(",")) + 1
            for index in range(layers):
                state[f"{name}.{2 * index}.weight"] = torch.from_numpy(
                    archive[f"{name}.{index}.w"].T.copy()
                )
                state[f"{name}.{2 * index}.bias"] = torch.from_numpy(
                    archive[f"{name}.{index}.b"]
                )
        for norm in ("obs_norm", "goal_norm"):
            for field in ("mean", "var", "count"):
                state[f"{norm}.{field}"] = torch.from_numpy(archive[f"{norm}.{field}"])
    state["ref_q0"] = torch.tensor(robot.default_joint_pos_list(), dtype=torch.float32)
    state["ref_alpha"] = torch.tensor(robot.action_scale_list(), dtype=torch.float32)
    return {
        "pi": state,
        "args": {
            **{
                key: args[key]
                for key in ("hist", "goal_frames", "enc_hidden", "dec_hidden", "pi_ema")
            },
            "robot": robot.name,
            "pi_ref_skip": True,
            "amp": "bf16",
        },
    }


def export_policy(checkpoint_path, output_dir, robot):
    checkpoint = read_checkpoint(checkpoint_path, robot)
    wrapper = load_policy(checkpoint, robot)
    metadata = contract(wrapper, checkpoint, robot)
    metadata.update(
        algorithm="xmimic",
        xmimic_upstream=(UPSTREAM / "UPSTREAM").read_text().strip(),
        checkpoint_sha256=sha256(checkpoint_path),
        weights_key="X-Mimic averaged policy arrays",
    )
    path = write_policy_bundle(
        wrapper,
        metadata,
        output_dir,
        robot.num_joints,
        "Policy trained with Nico Bohlinger's X-Mimic (MIT).\n"
        "https://github.com/nico-bohlinger/xmimic\n"
        "Exported through Cyclo MJLab and the BlinkMimic policy module.\n",
        {
            "Apache-2.0.txt": ROOT / "LICENSE",
            "XMimic-MIT.txt": UPSTREAM / "LICENSE",
            "XMimic-THIRD_PARTY_NOTICES.md": UPSTREAM / "THIRD_PARTY_NOTICES.md",
            "BlinkMimic-MIT.txt": ROOT / "third_party/blinkmimic/LICENSE",
            "BlinkMimic-THIRD_PARTY_NOTICES.md": ROOT
            / "third_party/blinkmimic/THIRD_PARTY_NOTICES.md",
        },
    )
    export_sim2real_cfg(output_dir, SimpleNamespace(**checkpoint["args"]), robot)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare()
    print(
        export_policy(
            args.checkpoint, args.output_dir, TASKS[args.task].robot_factory()
        )
    )


if __name__ == "__main__":
    main()
