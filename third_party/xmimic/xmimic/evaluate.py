"""Checkpoint evaluation with BlinkMimic's protocol (its tools/eval_batch.py), in JAX.

Every clip is played once from its first frame per replica, with friction 1.0, no randomization and no noise. A clip
fails if the training termination triggers. Normalized return = undiscounted reward sum / ((clip length - 1) *
MAX_STEP_REWARD). All checkpoints x clips x replicas step together in one batched MuJoCo Warp simulator (BlinkMimic's
constraint budget, njmax 500), with the policies applied by vmap over the checkpoint axis. On the same checkpoints the
mean return matches BlinkMimic's mjlab evaluator to +0.001.

    python -m xmimic.evaluate <run_dir> [--reps 4]

Writes <run_dir>/eval_jax_r<reps>.csv (BlinkMimic's columns) and prints the time to target.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import nets
from .kin import Kinematics
from .mdp import MAX_STEP_REWARD, MDP
from .model import load_model
from .motion import GOAL_Q_OFFSET, load_motion
from .sim import Sim


def load_ckpt(path):
    """(policy parameters, normalizers, metadata) of a checkpoint written by runtime.export_policy."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    params = {name: [{"w": z[f"{name}.{i}.w"], "b": z[f"{name}.{i}.b"]} for i in range(sum(1 for k in z.files if k.startswith(f"{name}.") and k.endswith(".w")))]
              for name in ("enc", "dec")}  # fmt: skip
    bufs = {n: {k: z[f"{n}.{k}"] for k in ("mean", "var", "count")} for n in ("obs_norm", "goal_norm")}
    return params, bufs, meta


def checkpoint_config(path):
    with np.load(path, allow_pickle=False) as checkpoint:
        return json.loads(checkpoint["meta"].item())["args"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reps", type=int, default=4, help="replicas of every clip")
    ap.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
    ap.add_argument("--chunk", type=int, default=500, help="control steps per compiled program")
    ap.add_argument("--motion_dir", default=None, help="default: the run's own")
    ap.add_argument("--group", type=int, default=0, help="checkpoints per batch (0: all at once)")
    ap.add_argument("--expected_robot", type=json.loads, default=None)
    args = ap.parse_args(argv)
    if min(args.reps, args.every, args.chunk) < 1 or args.group < 0:
        ap.error("reps, every, and chunk must be positive; group must be nonnegative")
    files = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.npz")))[:: args.every]
    if not files:
        ap.error(f"No checkpoints in {args.run_dir}")
    cfg = checkpoint_config(files[0])
    if args.expected_robot is not None and cfg.get("robot") != args.expected_robot:
        ap.error("Checkpoint robot differs from the selected task")
    keys = ("robot", "model_dir", "motion_dir", "hist", "goal_frames", "enc_hidden", "dec_hidden")
    for checkpoint in files[1:]:
        other = checkpoint_config(checkpoint)
        if any(cfg.get(key) != other.get(key) for key in keys):
            ap.error(f"Checkpoint differs from the selected run: {checkpoint}")
    mj, meta = load_model(cfg.get("model_dir"))
    if cfg.get("robot") != meta.get("robot"):
        ap.error("Checkpoint robot differs from the evaluation scene")
    kin = Kinematics(meta)
    feet = [meta["tracked_bodies"].index(b) for b in meta.get("contact_bodies", ("left_ankle_roll_link", "right_ankle_roll_link"))]
    motion = load_motion(args.motion_dir or cfg["motion_dir"], kin, feet)
    C = len(motion.names)
    K, Hh = cfg["goal_frames"], cfg["hist"]
    ref = (GOAL_Q_OFFSET, jnp.asarray(meta["default_joint_pos"], jnp.float32), jnp.asarray(meta["action_scale"], jnp.float32))
    groups = [files] if args.group <= 0 else [files[i : i + args.group] for i in range(0, len(files), args.group)]
    rows = []
    t_all = time.time()
    for grp in groups:
        cks = [load_ckpt(f) for f in grp]
        M, R = len(cks), args.reps
        N = M * R * C  # world n: checkpoint n // (R*C), replica (n // C) % R, clip n % C
        params = jax.tree.map(lambda *x: jnp.stack(x), *[c[0] for c in cks])
        bufs = jax.tree.map(lambda *x: jnp.stack(x), *[c[1] for c in cks])
        sim = Sim(mj, meta, N, None, eval_mode=True)
        mdp = MDP(kin, meta["default_joint_pos"], feet, sim.layout.dim)
        sim.mdp = mdp
        clip_of = jnp.arange(N) % C
        model_b, st = sim.init(jax.random.PRNGKey(0), motion)
        st = sim.reset_masked(motion, st, jnp.ones(N, bool), jax.random.PRNGKey(1), frames=motion.clip_start[clip_of])
        clip_end = motion.clip_end[clip_of]
        o = sim.observe(motion, st)
        carry = {"st": st, "hist": jnp.broadcast_to(o[:, None], (N, Hh, o.shape[-1])), "ret": jnp.zeros(N), "first": jnp.ones((), bool)}

        def act(p, b, h, g):
            return nets.pi_apply(p, b, h, g, ref)

        @jax.jit
        def run(carry, model_b, motion):
            def step(cr, _):
                st = cr["st"]
                o = sim.observe(motion, st)
                h = jnp.where(cr["first"], cr["hist"], jnp.concatenate((cr["hist"][:, 1:], o[:, None]), 1))
                g = motion.goal(st["f"], K)
                a = jax.vmap(act)(params, bufs, h.reshape(M, R * C, *h.shape[1:]), g.reshape(M, R * C, -1)).reshape(N, -1)
                alive = ~st["done_eval"]
                st, out = sim.step_core(model_b, motion, st, a)
                st = {**st, "f": jnp.minimum(st["f"], clip_end - 1)}
                return {"st": st, "hist": h, "ret": cr["ret"] + jnp.where(alive, out["r"], 0.0), "first": jnp.zeros((), bool)}, None

            carry, _ = jax.lax.scan(step, carry, None, length=args.chunk)
            return carry, carry["st"]["done_eval"].all()

        t0 = time.time()
        n_steps = 0
        while True:
            carry, all_done = run(carry, model_b, motion)
            n_steps += args.chunk
            if bool(all_done):
                break
        ret = np.asarray(carry["ret"]).reshape(M, R, C)
        failed = np.asarray(carry["st"]["failed_eval"]).reshape(M, R, C)
        clip_len = np.asarray(motion.clip_end - motion.clip_start + 1)
        norm = ret / ((clip_len - 1) * MAX_STEP_REWARD)
        for m, (f, (_, _, mt)) in enumerate(zip(grp, cks)):
            rows.append({"iteration": mt["iteration"], "train_wall_h": mt["wall_time"] / 3600, "norm_return": float(norm[m].mean()),
                         "norm_return_framew": float(ret[m].sum(-1).mean() / ((clip_len - 1).sum() * MAX_STEP_REWARD)),
                         "n_failed": float(failed[m].sum(-1).mean()),
                         "failed": "|".join(motion.names[i] for i in np.nonzero(failed[m].any(0))[0])})  # fmt: skip
            r = rows[-1]
            print(f"[eval] it {r['iteration']} ({r['train_wall_h'] * 60:.2f} min): return {r['norm_return']:.4f}, failed {r['n_failed']:.2f}/{C}", flush=True)
        print(f"group of {M} checkpoints x {R} reps x {C} clips = {N} worlds: {n_steps} steps in {time.time() - t0:.0f} s", flush=True)
    out = os.path.join(args.run_dir, f"eval_jax_r{args.reps}.csv")
    with open(out, "w") as fh:
        fh.write("iteration,train_wall_h,norm_return,norm_return_framew,n_failed,failed\n")
        for r in sorted(rows, key=lambda r: r["iteration"]):
            fh.write(f"{r['iteration']},{r['train_wall_h']:.6f},{r['norm_return']:.6f},{r['norm_return_framew']:.6f},{r['n_failed']},{r['failed']}\n")
    rows.sort(key=lambda r: r["iteration"])
    ok = [r["norm_return"] >= 0.82 for r in rows]
    strict = [r["norm_return"] >= 0.82 and r["n_failed"] == 0 for r in rows]
    first = next((r for r, o in zip(rows, ok) if o), None)
    held = next((rows[i] for i in range(len(rows) - 2) if all(strict[i : i + 3])), None)
    fmt = lambda r: "--" if r is None else f"{r['train_wall_h'] * 60:.2f} min (it {r['iteration']})"  # noqa: E731
    print(f"first >= 0.82: {fmt(first)} | >= 0.82 and 0 failed, held x3: {fmt(held)} | total eval {time.time() - t_all:.0f} s -> {out}")


if __name__ == "__main__":
    main()
