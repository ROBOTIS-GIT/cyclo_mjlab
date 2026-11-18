"""Batched G1 tracking simulator: MJX with the MuJoCo Warp backend, a port of blinkmimic/env.py (its --reset_masked
--bin_fix --fail_reset form). Resets are masked and fixed-size, nothing synchronizes with the host, and all state is a pytree.

Domain randomization is BlinkMimic's G1 default (dr.py): the contact friction of the robot geoms and a torso center-of-mass
offset once per world, joint encoder offsets once per episode. Together they form the sim context
xi = (friction, joint offsets, CoM offset), which the world model and the critic see.

A detail reproduced on purpose: BlinkMimic selects the ground as the geoms of body 0, but mjlab puts the plane on its own
body, so the selection is empty and the floor keeps friction 1.0. MuJoCo takes the larger friction of a contact pair, so the
robot's friction randomization U(0.3, 1.2) only acts within [1.0, 1.2]. XMimic changes only the robot geoms, as BlinkMimic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from .mdp import MAX_STEP_REWARD, flatten_state
from .motion import BIN_FRAMES
from .rot import mat_to_quat, mv3, quat_to_mat, so3_exp, tr3

DECIMATION = 4


class XiLayout:
    """Layout of the sim context xi = (friction 1, joint offsets J, CoM offset 3) and its normalization (center and
    half-range, as blinkmimic.dr.XiLayout)."""

    def __init__(self, meta: dict):
        J = len(meta["joint_names"])
        lo, hi = meta["dr"]["friction_range"]
        self.J = J
        self.dim = 1 + J + 3
        self.friction, self.joint_offset, self.com = slice(0, 1), slice(1, 1 + J), slice(1 + J, 4 + J)
        self.center = np.asarray([(lo + hi) / 2] + [0.0] * J + [0.0] * 3, np.float32)
        half = meta["dr"]["joint_offset"].get(".*", 0.01)
        self.offset_half = np.full((J,), max(half, 1e-6), np.float32)
        self.scale = np.asarray([(hi - lo) / 2] + [max(half, 1e-6)] * J + [0.05] * 3, np.float32)

    def normalize(self, xi):
        return (xi - self.center) / self.scale


class Sim:
    """``num_envs`` G1 worlds stepped together at 50 Hz (4 physics substeps of 5 ms per control step).

    Training mode samples reset frames from the reset bins (weighted by low return, recent failures and the probes' focus)
    and randomizes; ``eval_mode`` resets exactly onto the reference, without randomization and without timeouts."""

    def __init__(self, mj_model, meta: dict, num_envs: int, mdp, eval_mode: bool = False, gamma: float = 0.99,
                 max_episode_steps: int = 500, fail_reset: float = 0.0, device=None):  # fmt: skip
        from warp import JaxCallableGraphMode

        self.m, self.meta, self.N, self.mdp = mj_model, meta, num_envs, mdp
        self.eval_mode = eval_mode
        self.gamma, self.fail_reset = gamma, fail_reset
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.J = len(meta["joint_names"])
        self.layout = XiLayout(meta)
        self.qadr = np.asarray(meta["qpos_adr"])
        self.dadr = np.asarray(meta["dof_adr"])
        self.act = np.asarray(meta["actuator_of_joint"])
        self.fq, self.fd = meta["free_qpos_adr"], meta["free_dof_adr"]
        self.default_q = np.asarray(meta["default_joint_pos"], np.float32)
        self.alpha = np.asarray(meta["action_scale"], np.float32)
        self.robot_geoms = np.asarray(meta["robot_geom_ids"])
        self.com_body = meta["com_body_id"]
        self.com_range = np.asarray(meta["com_range"], np.float32)
        dr = meta["dr"]
        self.reset_pos, self.reset_rot = np.asarray(dr["reset_pos"], np.float32), np.asarray(dr["reset_rot"], np.float32)
        self.reset_lin_vel, self.reset_ang_vel = np.asarray(dr["reset_lin_vel"], np.float32), np.asarray(dr["reset_ang_vel"], np.float32)
        self.reset_joint_pos, self.reset_joint_vel = dr["reset_joint_pos"], dr["reset_joint_vel"]
        device = device if device is not None else jax.devices()[0]
        # Warp's own CUDA-graph capture of each physics call (graph mode "jax", XLA capturing the Warp call, is not supported)
        self.mx = mjx.put_model(mj_model, impl="warp", device=device, graph_mode=JaxCallableGraphMode.WARP)
        self.dx0 = mjx.make_data(mj_model, impl="warp", device=device, naconmax=num_envs * meta["sim"]["nconmax"], njmax=meta["sim"]["njmax"])
        self.model_axes = jax.tree.map(lambda _: None, self.mx).replace(geom_friction=0, body_ipos=0)

    # ------------------------------------------------------------------ setup
    def init(self, key, motion):
        """Batched model (per-env friction and CoM), initial env state, every world reset onto the reference."""
        N, L = self.N, self.layout
        kf, kc, kr = jax.random.split(key, 3)
        xi = jnp.zeros((N, L.dim))
        if self.eval_mode:
            fric = jnp.ones((N, 1))
            com = jnp.zeros((N, 3))
        else:
            lo, hi = self.meta["dr"]["friction_range"]
            fric = jax.random.uniform(kf, (N, 1), minval=lo, maxval=hi)
            com = jax.random.uniform(kc, (N, 3), minval=self.com_range[:, 0], maxval=self.com_range[:, 1])
        xi = xi.at[:, L.friction].set(fric).at[:, L.com].set(com)
        fr = jnp.broadcast_to(self.mx.geom_friction, (N,) + self.mx.geom_friction.shape)
        fr = fr.at[:, self.robot_geoms, 0].set(fric)
        ip = jnp.broadcast_to(self.mx.body_ipos, (N,) + self.mx.body_ipos.shape)
        ip = ip.at[:, self.com_body].add(com)
        model_b = self.mx.replace(geom_friction=fr, body_ipos=ip)
        nb = motion.num_bins
        data = jax.vmap(lambda q: self.dx0.replace(qpos=q))(jnp.tile(self.dx0.qpos[None], (N, 1)))
        st = {
            "data": data, "xi": xi,
            "f": jnp.zeros(N, jnp.int32), "ep_step": jnp.zeros(N, jnp.int32), "a_prev": jnp.zeros((N, self.J)),
            "start_bin": jnp.zeros(N, jnp.int32), "ep_disc_ret": jnp.zeros(N), "ep_ret": jnp.zeros(N),
            "bin_G": jnp.zeros(nb), "bin_w_avg": jnp.full(nb, 1.0 / nb, jnp.float32), "bin_fail": jnp.zeros(nb),
            "focus_q": jnp.full(nb, 1.0 / nb, jnp.float32), "focus_share": jnp.zeros((), jnp.float32),
            "done_eval": jnp.zeros(N, bool), "failed_eval": jnp.zeros(N, bool),
        }  # fmt: skip
        prev = np.arange(nb) - 1
        bin_clip = np.asarray(motion.bin_clip)
        self.bin_prev = np.asarray(np.where((prev >= 0) & (bin_clip[np.maximum(prev, 0)] == bin_clip), prev, prev + 1))
        st = self.reset_masked(motion, st, jnp.ones(N, bool), kr)
        return model_b, st

    # ------------------------------------------------------------------ state
    def state(self, data):
        qpos, qvel = data.qpos, data.qvel
        R = quat_to_mat(qpos[:, self.fq + 3 : self.fq + 7])
        return (qpos[:, self.fq : self.fq + 3], R, qvel[:, self.fd : self.fd + 3], mv3(R, qvel[:, self.fd + 3 : self.fd + 6]),
                qpos[:, self.qadr], qvel[:, self.dadr])  # fmt: skip

    # ------------------------------------------------------------------ reset
    def sample_frames(self, motion, st, key, gate):
        """Reset frames: a bin from BlinkMimic's mixture (uniform, low-return softmax, its running average, recent
        failures, and the probes' focus distribution), then a uniform frame inside the bin."""
        nb = motion.num_bins
        G_max = MAX_STEP_REWARD / (1 - self.gamma)
        soft = jax.nn.softmax(-25.0 * st["bin_G"] / G_max)
        w_avg = jnp.where(gate, st["bin_w_avg"] * 0.999 + 0.001 * soft, st["bin_w_avg"])
        probs = 0.5 / nb + 0.4 * soft + 0.1 * w_avg / w_avg.sum()
        if self.fail_reset > 0:
            fail = jnp.zeros(nb).at[self.bin_prev].add(st["bin_fail"])
            fail = fail / jnp.maximum(fail.sum(), 1e-6)
            probs = (1 - self.fail_reset) * probs + self.fail_reset * jnp.where(fail.sum() > 0, fail, probs)
        probs = (1 - st["focus_share"]) * probs + st["focus_share"] * st["focus_q"]
        kb, kf = jax.random.split(key)
        bins = jax.random.categorical(kb, jnp.log(jnp.maximum(probs, 1e-30)), shape=(self.N,))
        start = motion.bin_start[bins]
        end = jnp.minimum(start + BIN_FRAMES, motion.clip_end[motion.bin_clip[bins]])
        f = start + (jax.random.uniform(kf, (self.N,)) * (end - start)).astype(jnp.int32)
        return f, bins, w_avg

    def reset_masked(self, motion, st, mask, key, frames=None):
        """Reset the worlds where mask is true onto reference frames (sampled, or ``frames``); others stay bitwise unchanged."""
        N, J = self.N, self.J
        ks = jax.random.split(key, 8)
        if frames is None:
            frames, bins, w_avg = self.sample_frames(motion, st, ks[0], mask.any())
        else:
            bins, w_avg = motion.frame_bin[frames], st["bin_w_avg"]
        p, R, v, w, q, qd = motion.ref_state(frames)
        if not self.eval_mode:
            u = lambda k, half: (2 * jax.random.uniform(k, (N, half.shape[0])) - 1) * half  # noqa: E731
            p = p + u(ks[1], self.reset_pos)
            R = jnp.einsum("nij,njk->nik", so3_exp(u(ks[2], self.reset_rot)), R)
            v = v + u(ks[3], self.reset_lin_vel)
            w = w + u(ks[4], self.reset_ang_vel)
            q = q + (2 * jax.random.uniform(ks[5], (N, J)) - 1) * self.reset_joint_pos
            if self.reset_joint_vel > 0:
                qd = qd + (2 * jax.random.uniform(ks[6], (N, J)) - 1) * self.reset_joint_vel
        q = jnp.clip(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        d, d1 = mask, mask[:, None]
        data = st["data"]
        qpos = data.qpos.at[:, self.fq : self.fq + 3].set(p).at[:, self.fq + 3 : self.fq + 7].set(mat_to_quat(R)).at[:, self.qadr].set(q)
        qvel = data.qvel.at[:, self.fd : self.fd + 3].set(v).at[:, self.fd + 3 : self.fd + 6].set(mv3(tr3(R), w)).at[:, self.dadr].set(qd)
        ctrl = data.ctrl.at[:, self.act].set(q)
        data = data.replace(qpos=jnp.where(d1, qpos, data.qpos), qvel=jnp.where(d1, qvel, data.qvel), ctrl=jnp.where(d1, ctrl, data.ctrl),
                            qacc_warmstart=jnp.where(d1, 0.0, data.qacc_warmstart), time=jnp.where(d, 0.0, data.time))  # fmt: skip
        xi = st["xi"]
        if not self.eval_mode:
            L = self.layout
            new = (2 * jax.random.uniform(ks[7], (N, J)) - 1) * L.offset_half
            xi = xi.at[:, L.joint_offset].set(jnp.where(d1, new, xi[:, L.joint_offset]))
        return {**st, "data": data, "xi": xi, "bin_w_avg": w_avg,
                "f": jnp.where(d, frames, st["f"]), "ep_step": jnp.where(d, 0, st["ep_step"]),
                "a_prev": jnp.where(d1, 0.0, st["a_prev"]), "start_bin": jnp.where(d, bins, st["start_bin"]),
                "ep_disc_ret": jnp.where(d, 0.0, st["ep_disc_ret"]), "ep_ret": jnp.where(d, 0.0, st["ep_ret"])}  # fmt: skip

    # ------------------------------------------------------------------ step
    def physics(self, model_b, data, target):
        """PD targets in, 4 physics substeps per world (per-world friction and CoM through the batched model)."""

        def one(model, d, tgt):
            d = d.replace(ctrl=d.ctrl.at[self.act].set(tgt))
            return jax.lax.fori_loop(0, DECIMATION, lambda _, dd: mjx.step(model, dd), d)

        return jax.vmap(one, in_axes=(self.model_axes, 0, 0))(model_b, data, target)

    def observe(self, motion, st, key=None):
        o = self.mdp.obs(motion, self.state(st["data"]), st["a_prev"], st["xi"], st["f"])
        return o if key is None else self.mdp.add_noise(key, o)

    def step_core(self, model_b, motion, st, a):
        """Apply the action, run 4 physics substeps, label the transition (BlinkMimic TrackingSim.step_core)."""
        S = self.state(st["data"])
        f, xi, a_prev, ep_step = st["f"], st["xi"], st["a_prev"], st["ep_step"]
        target = self.default_q + xi[:, self.layout.joint_offset] + self.alpha * a
        data = self.physics(model_b, st["data"], target)
        S1 = self.state(data)
        f1 = f + 1
        r = self.mdp.reward(motion, S1, f1, a, a_prev)
        term = self.mdp.termination(motion, S1, f1)
        timeout = (f1 >= motion.clip_end[motion.clip_id[f]]) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {"s": flatten_state(S), "a": a, "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r, "term": term,
               "done": done, "ep_step": ep_step, "xi": xi, "cut": done}  # fmt: skip
        st = {**st, "data": data, "f": f1, "ep_step": ep_step + 1, "a_prev": a,
              "ep_disc_ret": st["ep_disc_ret"] + self.gamma ** ep_step.astype(jnp.float32) * r, "ep_ret": st["ep_ret"] + r}  # fmt: skip
        if self.eval_mode:
            st["failed_eval"] = st["failed_eval"] | (term & ~st["done_eval"])
            st["done_eval"] = st["done_eval"] | done
        return st, out

    def step_tail(self, motion, st, out, key):
        """Bin statistics of the ended episodes and masked resets (BlinkMimic step_tail_masked, --bin_fix)."""
        done, term, f = out["done"], out["term"], out["f"]
        dw = done.astype(jnp.float32)
        T = st["ep_step"].astype(jnp.float32)
        boot = jnp.where(term, 0.0, self.gamma**T * (st["ep_ret"] / T) / (1 - self.gamma))
        G = st["ep_disc_ret"] + boot
        st = {**st, "bin_G": _bin_ema_w(st["bin_G"], st["start_bin"], G, dw, 0.8)}
        if self.fail_reset > 0:
            fb = motion.frame_bin[jnp.minimum(f, motion.num_frames - 1)]
            bf = jnp.where(done.any(), st["bin_fail"] * 0.999, st["bin_fail"])
            st["bin_fail"] = _bin_ema_w(bf, fb, term.astype(jnp.float32), dw, 0.9)
        stats = {"n_done": dw.sum(), "ep_len_sum": (T * dw).sum(), "ep_ret_norm_sum": (st["ep_ret"] / (T * MAX_STEP_REWARD) * dw).sum(),
                 "term_sum": term.astype(jnp.float32).sum()}  # fmt: skip
        return self.reset_masked(motion, st, done, key), stats


def _bin_ema_w(stat, bins, x, wt, old: float):
    """Per-bin exponential average of x over the weighted samples that fall into each bin; bins without samples keep stat."""
    sums = jnp.zeros_like(stat).at[bins].add(x * wt)
    cnt = jnp.zeros_like(stat).at[bins].add(wt)
    return jnp.where(cnt > 0, old * stat + (1 - old) * sums / jnp.maximum(cnt, 1), stat)
