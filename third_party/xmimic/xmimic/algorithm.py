"""BlinkMimic's training iteration in JAX: the reference implementation of XMimic.

One iteration:
  1. collect: about 1 / utd simulation steps of all worlds with the current policy (MuJoCo Warp), into the replay buffer;
  2. world model: ``wm_batches`` updates on 8-step windows from the buffer, half of each batch mirrored left-right;
  3. policy: backpropagation through ``horizon`` imagined steps of the world model from buffer states (Muon, policy EMA);
  4. critic: GAE(lambda) targets on the imagined transitions, a few epochs of regression;
  plus BlinkMimic's progress-triggered policy-LR decay and its evaluation-guided resets (probe sweeps).

Everything is a pure function of a ``carry`` pytree, so ``make_chunk`` can run ``log_every`` iterations as one compiled XLA
program. xmimic/fast.py computes the same algorithm with fewer, larger GPU kernels; this file is the readable version.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import buffer as rb
from . import nets, optim
from .config import Config
from .kin import Kinematics
from .mdp import MDP, flatten_state, inv_dim, obs_dim, state_dim, unflatten_state
from .model import load_model
from .motion import CONTROL_DT, GOAL_Q_OFFSET, load_motion
from .sim import Sim

# floors of the world model's per-dimension error scale: position, rotation, linear and angular velocity (then joints)
ERR_FLOOR_PARTS = ((3, 3e-3), (3, 5e-3), (3, 2e-2), (3, 5e-2))
STAT_KEYS = ("r", "n_done", "ep_len_sum", "ep_ret_norm_sum", "term_sum")  # collection statistics, summed per iteration


def widths(s: str):
    return tuple(int(x) for x in s.split(","))


def flat32(tree):
    """All leaves of ``tree`` as one fresh float32 vector (never an alias of a carry leaf, which the next program donates)."""
    return jnp.concatenate([x.astype(jnp.float32).ravel() for x in jax.tree.leaves(tree)])


class Mirror:
    """Sagittal mirror M = diag(1, -1, 1) of states, joints (left/right swapped, roll and yaw joints negated) and the sim
    context (blinkmimic/mirror.py): world-model data augmentation."""

    def __init__(self, joint_names, layout):
        names = list(joint_names)
        perm, sign = [], []
        for n in names:
            partner = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            perm.append(names.index(partner))
            sign.append(-1.0 if any(f"_{t}_" in n for t in ("roll", "yaw")) else 1.0)
        self.perm, self.sign = np.asarray(perm), np.asarray(sign, np.float32)
        self.M = np.asarray([1.0, -1.0, 1.0], np.float32)
        self.layout = layout

    def joints(self, x):
        return x[..., self.perm] * self.sign

    def state(self, S):
        p, R, v, w, q, qd = S
        M = self.M
        return p * M, R * M[:, None] * M[None, :], v * M, -(w * M), self.joints(q), self.joints(qd)

    def xi(self, xi):
        L = self.layout
        xi = xi.at[..., L.joint_offset].set(self.joints(xi[..., L.joint_offset]))
        return xi.at[..., L.com.start + 1].set(-xi[..., L.com.start + 1])


class XMimic:
    """The model, the motion library, the simulators and every step of the training iteration as jittable functions."""

    def __init__(self, cfg: Config, probes: bool | None = None):
        """``probes``: build the in-loop probe simulator (default: whenever cfg.eprobe_share > 0). The multi-GPU layouts
        build the trainer without it and run the probe sweeps elsewhere."""
        self.cfg = c = cfg
        nets.MM_OUT = jnp.bfloat16  # torch-autocast matmul outputs (see nets.MM_OUT)
        self.mj_model, self.meta = load_model(c.model_dir)
        assert c.njmax > 0, "--njmax must be positive (BlinkMimic: 500)"
        self.meta["sim"]["njmax"] = c.njmax
        meta = self.meta
        self.J = J = len(meta["joint_names"])
        self.kin = Kinematics(meta)
        feet = [meta["tracked_bodies"].index(b) for b in meta.get("contact_bodies", ("left_ankle_roll_link", "right_ankle_roll_link"))]
        self.sim = Sim(self.mj_model, meta, c.num_envs, None, gamma=c.gamma, fail_reset=c.fail_reset)
        self.layout = self.sim.layout
        self.mdp = MDP(self.kin, meta["default_joint_pos"], feet, self.layout.dim)
        self.sim.mdp = self.mdp
        self.motion = load_motion(c.motion_dir, self.kin, feet)
        self.goal_dim = self.motion.goal_dim(c.goal_frames)
        self.obs_dim = obs_dim(J)
        self.critic_in = inv_dim(J) + self.mdp.critic_extra + self.goal_dim
        self.wm_in = inv_dim(J) + J + self.layout.dim
        self.wm_out = 12 + 2 * J
        self.ref = (GOAL_Q_OFFSET, np.asarray(meta["default_joint_pos"], np.float32), np.asarray(meta["action_scale"], np.float32))
        self.mirror = Mirror(meta["joint_names"], self.layout)
        self.err_floor = np.concatenate([np.full((n,), v, np.float32) for n, v in ERR_FLOOR_PARTS]
                                        + [np.full((J,), 3e-3, np.float32), np.full((J,), 5e-2, np.float32)])  # fmt: skip
        self.n_mirror = int(c.wm_batch * c.wm_mirror)
        self.probes = c.eprobe_share > 0 if probes is None else probes
        self.make_probe_tiles()
        if self.probes:
            self.penv = Sim(self.mj_model, meta, self.P, self.mdp, eval_mode=True, gamma=c.gamma)

    def make_probe_tiles(self):
        """Probe worlds: one per tile of ``eprobe_tile`` frames of every clip (ep_tile: each tile's first frame), and the
        next bin of every reset bin within its clip (ep_next) for spreading failures backwards."""
        c, mo = self.cfg, self.motion
        cs, ce = np.asarray(mo.clip_start), np.asarray(mo.clip_end)
        self.ep_tile = np.concatenate([np.arange(cs[i], ce[i] - 1, c.eprobe_tile) for i in range(len(cs))]).astype(np.int32)
        self.P = int(self.ep_tile.shape[0])
        nb = mo.num_bins
        nxt = np.arange(nb) + 1
        bc = np.asarray(mo.bin_clip)
        self.ep_next = np.asarray(np.where((nxt < nb) & (bc[np.minimum(nxt, nb - 1)] == bc), nxt, nxt - 1))

    # ------------------------------------------------------------------ state
    def init(self, key):
        """(consts, carry): the batched simulation model and motion library, and every piece of training state."""
        c, J = self.cfg, self.J
        ks = jax.random.split(key, 6)
        model_b, env = self.sim.init(ks[0], self.motion)
        pi, pi_b = nets.pi_init(ks[1], self.obs_dim, self.goal_dim, J, c.hist, enc_hidden=widths(c.enc_hidden),
                                dec_hidden=widths(c.dec_hidden))  # fmt: skip
        wm, wm_b = nets.wm_init(ks[2], self.wm_in, self.wm_out, widths(c.wm_hidden))
        critic, critic_b = nets.critic_init(ks[3], self.critic_in)
        self.pi_mask = {"enc": nets.muon_mask(pi["enc"]), "dec": nets.muon_mask(pi["dec"])}
        self.wm_mask = {"net": nets.muon_mask(wm["net"])}
        self.critic_mask = jax.tree.map(lambda _: False, critic)
        buf = rb.init(c.buffer_steps, c.num_envs, state_dim(J), J, self.layout.dim)
        carry = {
            "env": env, "obs_hist": jnp.zeros((c.num_envs, c.hist, self.obs_dim)), "buf": buf,
            "pi": pi, "pi_b": pi_b, "avg": jax.tree.map(jnp.copy, (pi, pi_b)), "avg_n": jnp.zeros((), jnp.int32),
            "wm": wm, "wm_b": wm_b, "critic": critic, "critic_b": critic_b,
            "opt_pi": optim.init(pi, self.pi_mask), "opt_wm": optim.init(wm, self.wm_mask), "opt_v": optim.init(critic, self.critic_mask),
            "key": ks[4], "it": jnp.zeros((), jnp.int32), "step_acc": jnp.zeros(()), "decay_at": jnp.full((), -1, jnp.int32),
            "dec_nd": jnp.zeros((), jnp.float32), "dec_rs": jnp.zeros((), jnp.float32),
            "nonfinite": jnp.zeros((), jnp.int32), "sim_steps": jnp.zeros((), jnp.float32),
        }  # fmt: skip
        consts = {"motion": self.motion, "model": model_b}
        if self.probes:
            model_p, pst = self.penv.init(ks[5], self.motion)
            consts["model_p"] = model_p
            P = self.P
            carry["ep"] = {"pst": pst, "hist": jnp.zeros((P, c.hist, self.obs_dim)), "fail_f": jnp.full((P,), -1, jnp.int32),
                           "stop": jnp.zeros((P,), jnp.int32), "fcap": jnp.zeros((P,), jnp.int32), "left": jnp.zeros((), jnp.int32),
                           "pending": jnp.zeros((), bool), "snap": jax.tree.map(jnp.copy, carry["avg"]),
                           "ep_fail": jnp.zeros(self.motion.num_bins), "n_fail": jnp.zeros((), jnp.int32)}  # fmt: skip
        # no weakly typed leaves: a jitted program returns strong types, and a weak input would force a second compile
        strong = lambda x: jnp.asarray(x, dtype=x.dtype) if isinstance(x, jax.Array) and x.weak_type else x  # noqa: E731
        return jax.tree.map(strong, consts), jax.tree.map(strong, carry)

    # ------------------------------------------------------------------ 1. collection
    def collect_step(self, consts, carry, key):
        """One control step of every world with the current policy (plus action noise); the transitions go to the replay
        buffer and update the observation, goal and world-model normalizers; ended episodes reset."""
        c, sim, motion = self.cfg, self.sim, consts["motion"]
        k_noise, k_act, k_tail = jax.random.split(key, 3)
        env = carry["env"]
        o = sim.observe(motion, env, k_noise)
        fresh = env["ep_step"] == 0
        oh = carry["obs_hist"]
        oh = jnp.where(fresh[:, None, None], jnp.broadcast_to(o[:, None], oh.shape), jnp.concatenate((oh[:, 1:], o[:, None]), 1))
        g = motion.goal(env["f"], c.goal_frames)
        pi_b = {"obs_norm": nets.rn_update(carry["pi_b"]["obs_norm"], o), "goal_norm": nets.rn_update(carry["pi_b"]["goal_norm"], g)}
        a = nets.pi_apply(carry["pi"], pi_b, oh, g, self.ref)
        a = jnp.clip(a + c.action_noise * jax.random.normal(k_act, a.shape), -20, 20)
        env, out = sim.step_core(consts["model"], motion, env, a)
        buf = rb.add(carry["buf"], out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wb = carry["wm_b"]
        wm_b = {"in_norm": nets.rn_update(wb["in_norm"], nets.wm_inputs(S0, out["a"], self.layout.normalize(out["xi"]))),
                "out_norm": nets.rn_update(wb["out_norm"], nets.wm_targets(S0, S1, CONTROL_DT)),
                "inc_norm": nets.rn_update(wb["inc_norm"], nets.increments(S0, S1))}  # fmt: skip
        env, tail = sim.step_tail(motion, env, out, k_tail)
        stats = {"r": out["r"].mean() / 0.1, **tail}
        return {**carry, "env": env, "obs_hist": oh, "buf": buf, "pi_b": pi_b, "wm_b": wm_b,
                "sim_steps": carry["sim_steps"] + c.num_envs}, stats  # fmt: skip

    def collect(self, consts, carry, key, n):
        """``n`` collection steps (``n`` traced: a while loop; static: the loop bound is known at compile time) and the
        summed collection statistics."""
        zero = {k: jnp.zeros((), jnp.float32) for k in STAT_KEYS}

        def body(i, cs):  # a loop with n trips (a cond around the step would copy the replay buffer)
            cr, acc = cs
            cr, s = self.collect_step(consts, cr, jax.random.fold_in(key, i))
            return cr, jax.tree.map(lambda x, y: x + y, acc, s)

        return jax.lax.fori_loop(0, n, body, (carry, zero))

    # ------------------------------------------------------------------ 2. world model
    def wm_loss(self, wm, wm_b, w):
        """Open-loop prediction over each window, error in the true state's frame, normalized per dimension."""
        c = self.cfg
        S = unflatten_state(w["s"][:, 0])
        xi_n = self.layout.normalize(w["xi"])
        preds = []
        for j in range(c.wm_window):
            S = nets.wm_step(wm, wm_b, S, w["a"][:, j], xi_n[:, j], CONTROL_DT)
            preds.append(S)
        S_hat = tuple(jnp.stack([P[i] for P in preds], 1) for i in range(6))
        std = jnp.maximum(nets.rn_std(wm_b["inc_norm"], 1e-4), self.err_floor)
        e = ((nets.state_error(S_hat, unflatten_state(w["s1"])) / std) ** 2).mean(-1)
        return (e * w["mask"]).sum() / w["s"].shape[0], e[:, -1]

    def wm_update(self, carry, key, lr):
        c, mir, n = self.cfg, self.mirror, self.n_mirror
        w = rb.sample_windows(carry["buf"], key, c.wm_batch, c.wm_window)
        if n > 0:
            ms = lambda x: flatten_state(mir.state(unflatten_state(x)))  # noqa: E731
            for k_, fn in (("s", ms), ("s1", ms), ("a", mir.joints), ("a_prev", mir.joints), ("xi", mir.xi)):
                w[k_] = jnp.concatenate((fn(w[k_][:n]), w[k_][n:]))
        (loss, e_last), grads = jax.value_and_grad(self.wm_loss, has_aux=True)(carry["wm"], carry["wm_b"], w)
        grads, _ = optim.clip_by_global_norm(grads, 1.0)
        wm, opt = optim.step(carry["wm"], grads, carry["opt_wm"], self.wm_mask, lr, betas=(0.9, 0.999))
        m = w["mask"][:, -1]
        return {**carry, "wm": wm, "opt_wm": opt}, {"wm": loss / c.wm_window, "wm8": (e_last * m).sum() / jnp.maximum(m.sum(), 1)}

    def wm_updates(self, carry, k_w, lr):
        """The iteration's ``wm_batches`` world-model updates."""
        return jax.lax.scan(lambda cr, k: self.wm_update(cr, k, lr), carry, jax.random.split(k_w, self.cfg.wm_batches))

    # ------------------------------------------------------------------ 3. policy and 4. critic
    def sample_valid_starts(self, motion, buf, key, B):
        """B imagination starts whose ``horizon`` reference frames stay inside their clip (drawn from 2B candidates)."""
        H = self.cfg.horizon
        s, h = rb.sample_starts(buf, key, 2 * B, self.cfg.hist)
        ok = s["f"] + H <= motion.clip_end[motion.clip_id[s["f"]]]
        idx = jnp.argsort((~ok).astype(jnp.int32), stable=True)[:B]
        return jax.tree.map(lambda x: x[idx], s), jax.tree.map(lambda x: x[idx], h)

    def imagine(self, pi, pi_b, wm, wm_b, critic, critic_b, motion, start, o_hist0, keys):
        """Roll the policy through the world model for ``horizon`` steps; the loss is minus the discounted imagined return
        (soft terminations, critic bootstrap at the end). Returns (loss, the rollout for the critic phase)."""
        c, mdp, H, K = self.cfg, self.mdp, self.cfg.horizon, self.cfg.goal_frames
        S0, xi, f0, a_prev0 = unflatten_state(start["s"]), start["xi"], start["f"], start["a_prev"]
        xi_n = self.layout.normalize(xi)
        F = f0[None] + jnp.arange(H, dtype=f0.dtype)[:, None]  # reference frame of each imagined step (H, B)
        G = motion.goal(F, K)
        Z = nets.pi_encode(pi, pi_b, G)  # the goal encodings of all steps in one batch (InstantTrack --batch_goal_enc)

        def step(carry, x):
            S, oh, a_prev, f = carry
            z, g, k = x
            a = nets.pi_apply(pi, pi_b, oh, g, self.ref, z=z)
            S1 = nets.wm_step(wm, wm_b, S, a, xi_n, CONTROL_DT)
            f1 = f + 1
            o1 = mdp.add_noise(k, mdp.obs(motion, S1, a, xi, f1))
            return (S1, jnp.concatenate((oh[:, 1:], o1[:, None]), 1), a, f1), (S1, a, a_prev)

        if c.imag_remat == "dots":
            step = jax.checkpoint(step, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        elif c.imag_remat == "full":
            step = jax.checkpoint(step)
        (S_H, _, _, f_H), (S_all, A, Ap) = jax.lax.scan(step, (S0, o_hist0, a_prev0, f0), (Z, G, keys), unroll=True)
        F1 = F + 1
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        Sf = tuple(flat(x) for x in S_all)
        r = mdp.reward(motion, Sf, flat(F1), flat(A), flat(Ap), w_contact=c.contact_reward).reshape(H, -1)
        d = mdp.soft_termination(motion, Sf, flat(F1)).reshape(H, -1)
        g_H = motion.goal(f_H, K)
        v_boot = nets.critic_apply(critic, critic_b, jnp.concatenate((mdp.critic_features(motion, S_H, f_H, xi_n), g_H), -1))
        alive = jnp.cumprod(1 - d, 0)
        c_t = jnp.concatenate((jnp.ones_like(alive[:1]), alive[:-1]), 0)
        disc = c.gamma ** jnp.arange(H, dtype=jnp.float32)[:, None]
        J = (disc * c_t * r).sum(0) + c.gamma**H * alive[-1] * v_boot
        aux = {"S0": S0, "S_all": S_all, "F": F, "G": G, "g_H": g_H, "xi_n": xi_n, "r": r, "d": d, "c_t": c_t, "surv": alive[-1]}
        return -J.mean(), aux

    def sample_policy_batch(self, consts, buf, key):
        """Imagination starts and their noisy observation histories (rebuilt from stored states, as in BlinkMimic)."""
        mdp, motion = self.mdp, consts["motion"]
        k_s, k_n = jax.random.split(key)
        start, h = self.sample_valid_starts(motion, buf, k_s, self.cfg.pi_batch)
        return start, mdp.add_noise(k_n, mdp.obs(motion, unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"]))

    def policy_update(self, consts, carry, key, lr, pi_lr, batch=None):
        """BPTT policy update, policy EMA, then the critic phase. ``batch`` = (starts, o_hist0) sampled elsewhere (the
        multi-GPU learners get it from the actor); otherwise it is sampled from carry["buf"]."""
        c, mdp, motion, H = self.cfg, self.mdp, consts["motion"], self.cfg.horizon
        k_s, k_n, k_i, k_c = jax.random.split(key, 4)
        start, o_hist0 = batch if batch is not None else self.sample_policy_batch(consts, carry["buf"], k_s)
        keys = jax.random.split(k_i, H)
        (loss, aux), grads = jax.value_and_grad(self.imagine, has_aux=True)(
            carry["pi"], carry["pi_b"], carry["wm"], carry["wm_b"], carry["critic"], carry["critic_b"], motion, start, o_hist0, keys)
        grads, gn = optim.clip_by_global_norm(grads, 1.0)
        if c.pi_clip_value > 0:
            grads = jax.tree.map(lambda g: jnp.clip(g, -c.pi_clip_value, c.pi_clip_value), grads)
        finite = jnp.isfinite(gn)  # a non-finite update is skipped (BlinkMimic --no_sync)
        grads = jax.tree.map(lambda g: jnp.where(finite, g, 0.0), grads)
        pi, opt_pi = optim.step(carry["pi"], grads, carry["opt_pi"], self.pi_mask, pi_lr, betas=(c.pi_beta1, 0.95), ok=finite)
        avg = optim.ema(carry["avg"], (pi, carry["pi_b"]), c.pi_ema, carry["avg_n"] == 0)
        # critic: GAE(lambda) targets over the imagined transitions, then epochs x minibatches of MSE regression
        sg = jax.lax.stop_gradient
        S_c = tuple(jnp.concatenate((sg(aux["S0"][i])[None], sg(aux["S_all"][i])), 0) for i in range(6))
        f_c = jnp.concatenate((aux["F"][:1], aux["F"] + 1), 0)
        g_c = jnp.concatenate((aux["G"], aux["g_H"][None]), 0)
        xi_c = jnp.broadcast_to(aux["xi_n"], (H + 1,) + aux["xi_n"].shape)
        X = jnp.concatenate((mdp.critic_features(motion, S_c, f_c, xi_c), g_c), -1)
        critic_b = {"norm": nets.rn_update(carry["critic_b"]["norm"], X)}
        rews, dones = sg(aux["r"]), sg(aux["d"])
        V = nets.critic_apply(carry["critic"], critic_b, X)

        def gae(R, t):
            R = rews[t] + c.gamma * (1 - dones[t]) * ((1 - c.lam) * V[t + 1] + c.lam * R)
            return R, R

        _, targets = jax.lax.scan(gae, V[-1], jnp.arange(H - 1, -1, -1))
        targets = targets[::-1]
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // c.critic_minibatches
        perms = jnp.concatenate([jax.random.permutation(k, n)[: mb * c.critic_minibatches].reshape(c.critic_minibatches, mb)
                                 for k in jax.random.split(k_c, c.critic_epochs)])  # fmt: skip

        def critic_step(vc, idx):
            params, opt = vc
            l, g = jax.value_and_grad(lambda p: ((nets.critic_apply(p, critic_b, Xf[idx]) - Yf[idx]) ** 2).mean())(params)
            g, _ = optim.clip_by_global_norm(g, 1.0)
            params, opt = optim.step(params, g, opt, self.critic_mask, lr)
            return (params, opt), l

        (critic, opt_v), v_losses = jax.lax.scan(critic_step, (carry["critic"], carry["opt_v"]), perms)
        stats = {"pi": loss, "v": v_losses[-1], "ir": ((aux["c_t"] * rews).sum(0) / H / 0.1).mean(), "gn": gn, "surv": aux["surv"].mean()}
        return {**carry, "pi": pi, "opt_pi": opt_pi, "avg": avg, "avg_n": carry["avg_n"] + 1, "critic": critic, "critic_b": critic_b,
                "opt_v": opt_v, "nonfinite": carry["nonfinite"] + (~finite).astype(jnp.int32)}, stats  # fmt: skip

    # ------------------------------------------------------------------ learning-rate schedule
    def lr_at(self, it):
        """Cosine schedule from lr to 3 % of it over lr_decay_iters (in practice nearly constant over 3500 iterations)."""
        c = self.cfg
        cos = c.lr * (0.03 + 0.97 * 0.5 * (1 + jnp.cos(jnp.pi * it / c.lr_decay_iters)))
        return jnp.where(it >= c.lr_decay_iters, c.lr * 0.03, cos)

    def pi_mult(self, it, decay_at):
        """The policy LR multiplier: pi_lr_mult until the decay trigger, then a linear ramp down to 1."""
        c = self.cfg
        ramp = c.pi_lr_mult + (1.0 - c.pi_lr_mult) * jnp.minimum(1.0, (it - decay_at) / max(c.lr_decay_len, 1))
        return jnp.where(decay_at >= 0, ramp, c.pi_lr_mult)

    def decay_tick(self, carry, it, cstats):
        """BlinkMimic's LR-decay trigger at the end of each log interval (it + 1 a multiple of log_every): the mean normalized
        return of the episodes finished in the interval reaches lr_decay_ret, or iteration lr_decay_by is reached. The new
        decay_at applies from the next iteration on, as when BlinkMimic's host loop sets it between intervals."""
        c = self.cfg
        nd, rs = carry["dec_nd"] + cstats["n_done"], carry["dec_rs"] + cstats["ep_ret_norm_sum"]
        it1 = it + 1
        end = it1 % c.log_every == 0
        hit = (nd > 0) & (rs / jnp.maximum(nd, 1.0) >= c.lr_decay_ret)
        if c.lr_decay_by > 0:
            hit = hit | (it1 >= c.lr_decay_by)
        trig = end & (carry["decay_at"] < 0) & hit & (c.lr_decay_ret > 0)
        zero = jnp.zeros((), jnp.float32)
        return {**carry, "decay_at": jnp.where(trig, it1, carry["decay_at"]), "dec_nd": jnp.where(end, zero, nd),
                "dec_rs": jnp.where(end, zero, rs)}  # fmt: skip

    # ------------------------------------------------------------------ evaluation-guided resets (probe sweeps)
    def eprobe_tick(self, consts, carry, it, key):
        """BlinkMimic's --eprobe_async in one iteration: apply a finished sweep's failures to the reset focus, start a sweep
        every eprobe_every iterations from eprobe_start, then advance the running sweep by up to eprobe_async steps."""
        carry = self.eprobe_control(consts, carry, it, key)
        return {**carry, "ep": self.eprobe_steps(consts, carry["ep"])}

    def eprobe_control(self, consts, carry, it, key):
        """Finish a sweep (failure counts -> reset focus of the training worlds) and start the next one."""
        c, motion, penv = self.cfg, consts["motion"], self.penv
        ep, env = carry["ep"], carry["env"]
        # finish: decayed first-failure counts per bin, spread over the eprobe_win bins before each failure
        fin = ep["pending"] & (ep["left"] == 0)
        ff = ep["fail_f"]
        ep_fail = ep["ep_fail"] * c.eprobe_decay
        ep_fail = ep_fail.at[motion.frame_bin[jnp.maximum(ff, 0)]].add((ff >= 0).astype(jnp.float32))
        w, q = ep_fail, ep_fail
        nb = motion.num_bins
        for _ in range(c.eprobe_win):
            w = w[self.ep_next] * (self.ep_next != jnp.arange(nb)).astype(jnp.float32)
            q = q + w
        has = fin & (ep_fail.sum() > 0)
        env = {**env, "focus_q": jnp.where(has, q / jnp.maximum(q.sum(), 1e-12), env["focus_q"]),
               "focus_share": jnp.where(has, c.eprobe_share, env["focus_share"])}  # fmt: skip
        ep = {**ep, "ep_fail": jnp.where(fin, ep_fail, ep["ep_fail"]), "pending": ep["pending"] & ~fin,
              "n_fail": jnp.where(fin, (ff >= 0).sum(), ep["n_fail"])}  # fmt: skip

        # start: a snapshot of the averaged policy; every probe resets on the reference a random lead before its tile
        def burst(ep):
            kl, kr = jax.random.split(key)
            lead = jax.random.randint(kl, (), c.eprobe_lead_min, c.eprobe_lead_max + 1)
            cid = motion.clip_id[self.ep_tile]
            f0 = jnp.maximum(self.ep_tile - lead, motion.clip_start[cid])
            stop = jnp.minimum(self.ep_tile + c.eprobe_tile, motion.clip_end[cid])
            fcap = jnp.minimum(stop, motion.clip_end[cid] - 1)
            pst = {**ep["pst"], "done_eval": jnp.zeros(self.P, bool), "failed_eval": jnp.zeros(self.P, bool)}
            pst = penv.reset_masked(motion, pst, jnp.ones(self.P, bool), kr, frames=f0)
            o = penv.observe(motion, pst)
            return {**ep, "pst": pst, "hist": jnp.broadcast_to(o[:, None], ep["hist"].shape), "fail_f": jnp.full((self.P,), -1, jnp.int32),
                    "stop": stop, "fcap": fcap, "left": lead + c.eprobe_tile, "snap": carry["avg"]}  # fmt: skip

        start = (it >= c.eprobe_start) & (it % c.eprobe_every == 0) & (ep["left"] == 0) & ~ep["pending"]
        ep = jax.lax.cond(start, burst, lambda e: e, ep)
        return {**carry, "ep": ep, "env": env}

    def eprobe_steps(self, consts, ep):
        """Up to eprobe_async steps of the running sweep, recording each probe's first failure. A probe's frame advances by
        one per step and stops at its cap (probes never reset during a sweep), so the goals of all these steps are known in
        advance and encoded as one batch."""
        c, motion, penv = self.cfg, consts["motion"], self.penv
        snap_p, snap_b = ep["snap"]
        K = c.eprobe_async
        k = jnp.minimum(K, ep["left"])

        def run(ep):
            F = jnp.minimum(ep["pst"]["f"][None] + jnp.arange(K, dtype=jnp.int32)[:, None], ep["fcap"][None])  # (K, P)
            G = motion.goal(F, c.goal_frames)
            Z = nets.pi_encode(snap_p, snap_b, G)

            def step(j, s):
                pst, hist, fail_f = s
                o = penv.observe(motion, pst)
                h = jnp.concatenate((hist[:, 1:], o[:, None]), 1)
                a = nets.pi_apply(snap_p, snap_b, h, G[j], self.ref, z=Z[j])
                alive = ~pst["done_eval"]
                f_now = pst["f"]
                pst, out = penv.step_core(consts["model_p"], motion, pst, a)
                failed = alive & (out["term"] | ~jnp.isfinite(a).all(-1))
                fail_f = jnp.where(failed, f_now, fail_f)
                pst = {**pst, "done_eval": pst["done_eval"] | (pst["f"] >= ep["stop"]), "f": jnp.minimum(pst["f"], ep["fcap"])}
                return pst, h, fail_f

            return jax.lax.fori_loop(0, k, step, (ep["pst"], ep["hist"], ep["fail_f"]))

        pst, hist, fail_f = jax.lax.cond(k > 0, run, lambda e: (e["pst"], e["hist"], e["fail_f"]), ep)
        left = ep["left"] - k
        return {**ep, "pst": pst, "hist": hist, "fail_f": fail_f, "left": left, "pending": ep["pending"] | ((k > 0) & (left == 0))}

    # ------------------------------------------------------------------ one iteration, and many as one program
    def iteration(self, consts, carry, _):
        """One training iteration (1GPU layout): probes, collection, LR-decay trigger, world model, policy and critic."""
        c = self.cfg
        it = carry["it"]
        lr = self.lr_at(it.astype(jnp.float32))
        pi_lr = lr * self.pi_mult(it.astype(jnp.float32), carry["decay_at"].astype(jnp.float32))
        key, k_c, k_w, k_p, k_e = jax.random.split(carry["key"], 5)
        acc = carry["step_acc"] + 1.0 / c.utd  # BlinkMimic's schedule: floor(acc) collection steps, acc keeps the rest
        n = jnp.floor(acc).astype(jnp.int32)
        carry = {**carry, "key": key, "step_acc": acc - n}
        if self.probes:
            carry = self.eprobe_tick(consts, carry, it, k_e)
        carry, cstats = self.collect(consts, carry, k_c, n)
        carry = self.decay_tick(carry, it, cstats)
        carry, wstats = self.wm_updates(carry, k_w, lr)
        carry, pstats = self.policy_update(consts, carry, k_p, lr, pi_lr)
        stats = {"wm": wstats["wm"][-1], "wm8": wstats["wm8"][-1], **pstats, "n_coll": jnp.asarray(n, jnp.float32),
                 **cstats, "lr": lr, "pi_lr": pi_lr, "decay_at": carry["decay_at"].astype(jnp.float32),
                 "sim_steps": carry["sim_steps"], "nonfinite": carry["nonfinite"].astype(jnp.float32)}  # fmt: skip
        if self.probes:
            stats["ep_n_fail"] = carry["ep"]["n_fail"].astype(jnp.float32)
            stats["ep_left"] = carry["ep"]["left"].astype(jnp.float32)
        return {**carry, "it": it + 1}, stats

    def make_chunk(self, n_iters: int):
        """``n_iters`` iterations as one program (lax.scan, the carry donated) -> (carry, per-iteration stats, the averaged
        policy as one fresh float32 vector for checkpoints and probe snapshots, readable while the next chunk runs)."""

        @partial(jax.jit, donate_argnums=(1,))
        def chunk(consts, carry):
            carry, st = jax.lax.scan(partial(self.iteration, consts), carry, None, length=n_iters)
            return carry, st, flat32(carry["avg"])

        return chunk

    def make_warmup(self, n_steps: int):
        """``n_steps`` collection steps with the initial policy before the first update."""

        @partial(jax.jit, donate_argnums=(1,))
        def warm(consts, carry):
            def body(cr, k):
                cr, s = self.collect_step(consts, cr, k)
                return cr, s

            key, k = jax.random.split(carry["key"])
            carry, s = jax.lax.scan(body, {**carry, "key": key}, jax.random.split(k, n_steps))
            return carry, s

        return warm
