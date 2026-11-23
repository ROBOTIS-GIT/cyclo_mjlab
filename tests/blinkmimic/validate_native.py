# ruff: noqa: E402  # Vendored path bootstrap must precede upstream imports.
"""Bounded native GPU diagnostic; full widths, one-step gradients.

This is not the full released training loop or a convergence result.
Run separately from training, on an idle GPU.
"""

import argparse
import json
from pathlib import Path

import torch

from source.tasks.blinkmimic.runtime import prepare

prepare()
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP, unflatten_state
from blinkmimic.networks import Policy, WorldModel, policy_kwargs
from blinkmimic.optim import MuonAdam

from source.tasks.blinkmimic.reference.motion import ExactMotionLib, verify_manifest
from source.tasks.blinkmimic.config.k1_rev1.robot_spec import make_robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dev = "cuda:0"
    torch.manual_seed(0)
    robot = make_robot()
    manifest = verify_manifest(
        args.motion_dir / "manifest.json", verify_sources=args.verify_sources
    )
    kin = Kinematics(robot, dev)
    motion = ExactMotionLib(manifest, robot, kin, dev)
    layout = XiLayout(robot, dev)
    mdp = MDP(
        kin,
        motion,
        torch.tensor(robot.default_joint_pos_list(), device=dev),
        layout.dim,
    )
    mdp.feet = [
        robot.tracked_bodies.index(f"{side}_ankle_roll_link")
        for side in ("left", "right")
    ]
    motion.set_ref_contact(mdp.feet)
    N, K, hist = len(motion.names), 10, 10
    env = TrackingSim(
        make_sim(robot), N, robot, motion, mdp, dev, bin_fix=True, fail_reset=0.3
    )
    env.reset(torch.arange(N, device=dev), motion.clip_start.clone())
    pi = Policy(
        mdp.obs_dim,
        motion.goal_dim(K),
        23,
        hist=hist,
        **policy_kwargs({"pi_ref_skip": True}, robot),
    ).to(dev)
    wm = WorldModel(23, layout.dim, hidden=(1024, 1024, 1024), structured=True).to(dev)
    opt_wm = MuonAdam(wm, [wm.net], lr=4e-4)
    opt_pi = MuonAdam(pi, [pi.enc, pi.dec], lr=0.0016)
    start_wm = {k: v.detach().clone() for k, v in wm.named_parameters()}
    start_pi = {k: v.detach().clone() for k, v in pi.named_parameters()}
    with torch.no_grad():
        obs = env.observe()
        obs_hist = obs[:, None].repeat(1, hist, 1)
        pi.obs_norm.update(obs)
        pi.goal_norm.update(motion.goal(env.f, K))
        a = pi(obs_hist, motion.goal(env.f, K))
        transition = env.step(a)
    S = unflatten_state(transition["s"])
    target = unflatten_state(transition["s1"])
    xi = layout.normalize(transition["xi"])
    pred = wm.step(S, transition["a"], transition["a_prev"], xi)
    loss = wm.state_error(pred, target).square().mean()
    opt_wm.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in wm.parameters())
    opt_wm.step()
    wm_after = {k: v.detach().clone() for k, v in wm.named_parameters()}
    for p in wm.parameters():
        p.requires_grad_(False)
    a = pi(obs_hist, motion.goal(transition["f"], K))
    pred = wm.step(S, a, transition["a_prev"], xi)
    actor_loss = -mdp.reward(
        pred, transition["f"] + 1, a, transition["a_prev"], w_contact=0.25
    ).mean()
    opt_pi.zero_grad()
    actor_loss.backward()
    assert torch.isfinite(actor_loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in pi.parameters())
    opt_pi.step()
    assert all(torch.equal(wm_after[k], v) for k, v in wm.named_parameters())
    for _ in range(7):
        with torch.no_grad():
            obs = env.observe()
            obs_hist = torch.cat((obs_hist[:, 1:], obs[:, None]), 1)
            a = pi(obs_hist, motion.goal(env.f, K))
            assert torch.isfinite(a).all()
            out = env.step(a)
            assert torch.isfinite(out["s1"]).all()
    wm_delta = max(
        (v - start_wm[k]).abs().max().item() for k, v in wm.named_parameters()
    )
    pi_delta = max(
        (v - start_pi[k]).abs().max().item() for k, v in pi.named_parameters()
    )
    assert wm_delta > 0 and pi_delta > 0
    path = args.output / "diagnostic.pt"
    torch.save(
        {"pi": pi.state_dict(), "wm": wm.state_dict(), "scope": "one-step diagnostic"},
        path,
    )
    ck = torch.load(path, weights_only=True)
    assert all(torch.equal(v, ck["pi"][k]) for k, v in pi.state_dict().items())
    assert all(torch.equal(v, ck["wm"][k]) for k, v in wm.state_dict().items())
    result = dict(
        status="PASS",
        clips=N,
        frames=motion.num_frames,
        initial_clip_starts=N,
        vector_steps=8,
        wm_hidden=[1024] * 3,
        encoder_decoder=[2048, 1024, 512, 256],
        history=hist,
        goal_frames=K,
        wm_loss=float(loss),
        actor_loss=float(actor_loss),
        wm_delta=wm_delta,
        pi_delta=pi_delta,
        wm_frozen_during_actor=True,
        checkpoint=str(path),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope="Bounded native diagnostic; one-step Muon updates, not official horizon-32 learner acceptance or convergence",
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
