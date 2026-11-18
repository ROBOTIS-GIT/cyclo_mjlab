"""The fast learner: XMimic's world-model update, BPTT policy update and critic fit with fewer, larger GPU kernels on
the sequential paths. Same algorithm and hyperparameters as the reference (xmimic/algorithm.py); tests/test_fast_equiv.py
checks each piece against it on the CPU.

What changes, and how close it stays to the reference (exact = bitwise; fp32 = same math, another fp32 rounding order;
bf16 = bf16 roundings in other places, well inside the reference's own bf16 error):

Policy update
- start sampling   exact  draw 2B candidate starts, but gather states and histories only for the B kept ones.
- imagination      fp32   everything that does not depend on the imagined state is hoisted out of the 32-step recurrence:
                          the observation noise of all steps in one batched draw, reference rotations, the ref-skip term
                          of the policy, the goal of the last step. Each observation is normalized once when produced
                          (the reference renormalizes the 10-frame history every step), and the critic bootstrap reuses
                          the reward's forward kinematics.
- contact reward   exact  feet heights and slips by static slices (an index list is a gather, whose transpose scatters).
- decoder grads    bf16   weight and bias gradients of the policy decoder as one bf16 GEMM per layer over all H*B rows
                          after the backward recurrence, instead of per-step GEMMs inside it.
- Muon             bf16   Newton-Schulz of same-shape matrices as batched GEMMs.
- critic           fp32   inputs reuse the imagination's kinematics, are normalized and cast to bf16 once; the 5
                          permutations come from one batched sort; GAE and the 20 regression steps are unrolled.
- GEMM outputs     bf16   every learner GEMM returns bf16 (torch autocast's bf16 Linear), so JAX's own transpose runs all
                          backward GEMMs in bf16 instead of fp32 / TF32.
- no remat         exact  imagination activations are stored, not recomputed in the backward pass (more memory).
World model
- presampling      exact  the windows of all wm_batches updates are sampled and mirrored in one pass before the update
                          loop; target rotations and the error scale are computed once per iteration.
- weight grads     bf16   one bf16 GEMM per layer over the 8*B rows.
- rotations        fp32   an elementwise so3_log in the loss (no argmax, no scatter-add in the backward pass).
- LayerNorm        fp32   a hand-written backward pass (fast_nets.layer_norm).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import buffer as rb
from . import fast_nets as fn
from . import nets, optim
from .algorithm import XMimic
from .mdp import BODY_ANGVEL_STD, BODY_LINVEL_STD, BODY_ORI_STD, BODY_POS_STD, CONTACT_Z, ROOT_ORI_STD, ROOT_POS_STD
from .mdp import W_ACTION_RATE, W_JOINT_LIMIT, flatten_state, invariant_features, unflatten_state
from .motion import CONTROL_DT
from .rot import mat_to_6d, mm3, mv3, rot_angle_sq, tr3, yaw_mat

BF16 = jnp.bfloat16
sg = jax.lax.stop_gradient


def keyed_map(f, keys):
    """[f(k) for k in keys] stacked, as one vmapped computation (vmapped threefry draws are bitwise the per-key draws)."""
    return jax.vmap(f)(keys)


class FastXMimic(XMimic):
    # ------------------------------------------------------------------ helpers
    def _mlp(self, layers, x, act, **kw):
        return fn.mlp_apply(layers, x, act, **kw)

    def _wm_step(self, wm, wm_b, S, a, xi_n, taps=None, defer=False):
        """nets.wm_step; returns (S1, the bf16 inputs of the world-model layers)."""
        y, ins = self._mlp(wm["net"], nets.rn_apply(wm_b["in_norm"], nets.wm_inputs(S, a, xi_n)), jax.nn.silu, taps=taps, defer=defer)
        return nets.wm_integrate(S, y * nets.rn_std(wm_b["out_norm"], 1e-4) + wm_b["out_norm"]["mean"], CONTROL_DT), ins

    def _critic_apply(self, params, x_n):
        """nets.critic_apply on inputs already normalized (and cast to bf16, which the first layer does anyway)."""
        return self._mlp(params["net"], x_n, jax.nn.elu)[0][..., 0]

    def _feet(self, x, comp):
        return jnp.stack([x[..., i, comp] for i in self.mdp.feet], -1)

    def _reward_fk(self, motion, S, f, a, a_prev, fk, w_contact):
        """mdp.reward from precomputed forward kinematics; also returns the anchored reference for the critic (exact)."""
        mdp = self.mdp
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = fk
        rp, rR, dR, ref_bp = mdp._anchored_ref(motion, p, R, f)
        ref_bR = mm3(dR[..., None, :, :], motion.body_R[f])
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((motion.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((motion.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (jnp.exp(-e_pos / BODY_POS_STD**2) + jnp.exp(-e_ori / BODY_ORI_STD**2) + jnp.exp(-e_lv / BODY_LINVEL_STD**2)
                   + jnp.exp(-e_av / BODY_ANGVEL_STD**2) + 0.5 * jnp.exp(-e_rp / ROOT_POS_STD**2) + 0.5 * jnp.exp(-e_ro / ROOT_ORI_STD**2))  # fmt: skip
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (jax.nn.relu(q - mdp.soft_hi) + jax.nn.relu(mdp.soft_lo - q)).sum(-1)
        r = (r_track + W_ACTION_RATE * r_rate + W_JOINT_LIMIT * r_lim) * CONTROL_DT
        if w_contact > 0:
            fz, slip = self._feet(bp, 2), (self._feet(bv, 0) ** 2 + self._feet(bv, 1) ** 2)
            c_sim = jax.nn.sigmoid((CONTACT_Z - fz) / 0.005)
            c_ref = motion.ref_contact[f]
            r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * jnp.exp(-slip / 0.3**2)
            r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
        return r, (rp, rR, ref_bp)

    def _critic_features_fk(self, motion, S, f, xi_n, bp, anch):
        """mdp.critic_features with the forward kinematics and anchored reference given (exact)."""
        p, R, v, w, q, qd = S
        rp, rR, ref_bp = anch
        Ht = tr3(yaw_mat(R))
        e_body = mv3(Ht[..., None, :, :], ref_bp - bp).reshape(bp.shape[:-2] + (-1,))
        e_root = mv3(Ht, rp - p)
        rel = mat_to_6d(mm3(tr3(R), rR))
        return jnp.concatenate((invariant_features(S), motion.q[f] - q, e_body, e_root, rel, xi_n), -1)

    # ------------------------------------------------------------------ policy batch
    def sample_valid_starts(self, motion, buf, key, B):
        """rb.sample_starts(buf, key, 2B, hist) and the stable partition of the reference, but every field is gathered
        for the kept rows only (exact)."""
        H, hist = self.cfg.horizon, self.cfg.hist
        ka, ke = jax.random.split(key)
        N = buf["s"].shape[1]
        age = rb._uniform_int(ka, buf["size"], 2 * B)
        env = jax.random.randint(ke, (2 * B,), 0, N)
        row = rb._row(buf, age)
        f = buf["f"][row, env]
        ok = f + H <= motion.clip_end[motion.clip_id[f]]
        idx = jnp.argsort((~ok).astype(jnp.int32), stable=True)[:B]
        age, env, row = age[idx], env[idx], row[idx]
        ep_step = buf["ep_step"][row, env]
        back = jnp.arange(hist - 1, -1, -1)
        back = jnp.minimum(back[None, :], ep_step[:, None])
        back = jnp.minimum(back, age[:, None])
        hrows = rb._row(buf, age[:, None] - back)
        henv = env[:, None]
        start = {k: buf[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: buf[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d

    # ------------------------------------------------------------------ imagination
    def _noise(self, keys, B):
        """mdp.add_noise's draws for every imagined step: one uniform(k, (B, D)) per step key, in one batch."""
        D = self.obs_dim
        u = keyed_map(lambda k: jax.random.uniform(k, (B, D)), keys)
        return (2 * u - 1) * self.mdp.obs_noise

    def imagine(self, pi, pi_b, wm, wm_b, critic, critic_b, motion, start, o_hist0, keys, taps=None):
        """XMimic.imagine with the hoisting described in the module docstring. ``taps`` (zeros per decoder layer): defer
        the decoder's weight gradients (``policy_grads``)."""
        c = self.cfg
        mdp, H, K, J, D = self.mdp, c.horizon, c.goal_frames, self.J, self.obs_dim
        defer = taps is not None
        S0, xi, f0, a_prev0 = unflatten_state(start["s"]), start["xi"], start["f"], start["a_prev"]
        B = f0.shape[0]
        xi_n = self.layout.normalize(xi)
        F_all = f0[None] + jnp.arange(H + 1, dtype=f0.dtype)[:, None]  # frames f0 .. f0+H
        F, F1 = F_all[:H], F_all[1:]
        G_all = motion.goal(F_all, K)
        G, g_H = G_all[:H], G_all[H]
        Z = jnp.tanh(self._mlp(pi["enc"], nets.rn_apply(pi_b["goal_norm"], G), jax.nn.elu)[0])  # nets.pi_encode
        off, q0, alpha = self.ref
        GR = (G[..., off : off + J] - q0) / alpha  # ref-skip term of pi_apply
        RR = motion.root_R[F1]  # reference root rotation in mdp.obs at frame f+1
        NZ = self._noise(keys, B)  # mdp.add_noise for every step
        dq0 = xi[..., 1 : 1 + J]
        hn0 = nets.rn_apply(pi_b["obs_norm"], o_hist0).reshape(B, -1)  # normalized history, oldest first
        on = pi_b["obs_norm"]

        def obs(S, a_prev, rr):  # mdp.obs (exact), reference rotation given
            p, R, v, w, q, qd = S
            Rt = tr3(R)
            return jnp.concatenate((q - mdp.default_q - dq0, qd, mv3(Rt, w), -R[..., 2, :], a_prev, mat_to_6d(mm3(Rt, rr))), -1)

        def step(carry, x):
            S, hn, a_prev = carry
            z, gr, rr, nz, tp = x
            a, ins = self._mlp(pi["dec"], jnp.concatenate((z, hn), -1), jax.nn.elu, taps=tp, defer=defer)
            a = a + gr
            S1, _ = self._wm_step(wm, wm_b, S, a, xi_n)
            o1 = obs(S1, a, rr) + nz
            hn1 = jnp.concatenate((hn[:, D:], nets.rn_apply(on, o1)), -1)
            return (S1, hn1, a), (S1, a, a_prev, ins if defer else None)

        if c.imag_remat == "dots":
            step = jax.checkpoint(step, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        elif c.imag_remat == "full":
            step = jax.checkpoint(step)
        xs = (Z, GR, RR, NZ, tuple(taps) if defer else None)
        (S_H, _, _), (S_all, A, Ap, dec_in) = jax.lax.scan(step, (S0, hn0, a_prev0), xs, unroll=True)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        Sf = tuple(flat(x) for x in S_all)
        F1f = flat(F1)
        fk = self.kin.forward(*Sf)
        r, anch = self._reward_fk(motion, Sf, F1f, flat(A), flat(Ap), fk, c.contact_reward)
        r = r.reshape(H, -1)
        d = mdp.soft_termination(motion, Sf, F1f).reshape(H, -1)
        # critic bootstrap on S_H from the same kinematics (the reference runs a second forward kinematics on S_H)
        unf = lambda x: x.reshape((H, -1) + x.shape[1:])  # noqa: E731
        bp_all = unf(fk[0])
        anch_all = tuple(unf(x) for x in anch)
        f_H = F_all[H]
        feat_H = self._critic_features_fk(motion, S_H, f_H, xi_n, bp_all[-1], tuple(x[-1] for x in anch_all))
        v_boot = self._critic_apply(critic, nets.rn_apply(critic_b["norm"], jnp.concatenate((feat_H, g_H), -1)))
        alive = jnp.cumprod(1 - d, 0)
        c_t = jnp.concatenate((jnp.ones_like(alive[:1]), alive[:-1]), 0)
        disc = c.gamma ** jnp.arange(H, dtype=jnp.float32)[:, None]
        Jr = (disc * c_t * r).sum(0) + c.gamma**H * alive[-1] * v_boot
        aux = {"S0": S0, "S_all": S_all, "F": F, "G": G, "g_H": g_H, "xi_n": xi_n, "r": r, "d": d, "c_t": c_t, "surv": alive[-1],
               "F_all": F_all, "G_all": G_all, "bp_all": sg(bp_all), "anch_all": jax.tree.map(sg, anch_all)}  # fmt: skip
        if defer:
            aux["dec_in"] = dec_in
        return -Jr.mean(), aux

    # ------------------------------------------------------------------ critic
    def critic_features(self, motion, aux):
        """The state part of the critic inputs of the H+1 states (start and imagined), from the imagination's kinematics:
        X = concat(features, goals of the frames aux["F_all"])."""
        H = self.cfg.horizon
        S0, S_all = tuple(sg(x) for x in aux["S0"]), tuple(sg(x) for x in aux["S_all"])
        xi_n = aux["xi_n"]
        feat0 = self.mdp.critic_features(motion, S0, aux["F"][0], xi_n)
        feat = self._critic_features_fk(motion, S_all, aux["F_all"][1:], jnp.broadcast_to(xi_n, (H,) + xi_n.shape),
                                        aux["bp_all"], aux["anch_all"])  # fmt: skip
        return jnp.concatenate((feat0[None], feat), 0)

    def critic_fit(self, carry, X, rews, dones, k_c, lr):
        """The critic phase on its inputs X (H+1, B, D) and the imagined rewards and soft terminations (H, B): normalizer
        update, GAE(lambda) targets, epochs x minibatches of regression. ``carry`` holds critic, critic_b and opt_v."""
        c, H = self.cfg, self.cfg.horizon
        critic_b = {"norm": nets.rn_update(carry["critic_b"]["norm"], X)}
        Xn = nets.rn_apply(critic_b["norm"], X).astype(BF16)  # what the first critic layer computes from every row
        V = self._critic_apply(carry["critic"], Xn)
        R, tg = V[-1], [None] * H
        for t in range(H - 1, -1, -1):  # the GAE recursion, unrolled (the same expression per step)
            R = rews[t] + c.gamma * (1 - dones[t]) * ((1 - c.lam) * V[t + 1] + c.lam * R)
            tg[t] = R
        targets = jnp.stack(tg)
        # minibatch rows are drawn from range(H*B): the first H*B rows of the flattened (H+1, B, D) inputs are X[:-1], so
        # gather from the whole array instead of materializing the X[:-1] slice
        Xf, Yf = Xn.reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Yf.shape[0]
        mb = n // c.critic_minibatches
        P = keyed_map(lambda k: jax.random.permutation(k, n), jax.random.split(k_c, c.critic_epochs))
        perms = P[:, : mb * c.critic_minibatches].reshape(c.critic_epochs * c.critic_minibatches, mb)

        def critic_step(vc, idx):
            params, opt = vc
            l, g = jax.value_and_grad(lambda p: ((self._critic_apply(p, Xf[idx]) - Yf[idx]) ** 2).mean())(params)
            g, _ = optim.clip_by_global_norm(g, 1.0)
            params, opt = optim.step(params, g, opt, self.critic_mask, lr)
            return (params, opt), l

        (critic, opt_v), v_losses = jax.lax.scan(critic_step, (carry["critic"], carry["opt_v"]), perms, unroll=True)
        return critic, critic_b, opt_v, v_losses

    # ------------------------------------------------------------------ policy update
    def policy_grads(self, carry, motion, start, o_hist0, keys):
        """(loss, aux, d loss / d pi) of the imagination, with the decoder's weight gradients deferred: bf16 zero taps on
        the decoder pre-activations collect the cotangents, ``fn.deferred_grads`` turns them into dW and db."""
        H = self.cfg.horizon
        B = start["f"].shape[0]
        args = (carry["pi_b"], carry["wm"], carry["wm_b"], carry["critic"], carry["critic_b"], motion, start, o_hist0, keys)
        taps0 = tuple(jnp.zeros((H, B, layer["w"].shape[1]), BF16) for layer in carry["pi"]["dec"])
        (loss, aux), (g_pi, g_taps) = jax.value_and_grad(lambda p, t: self.imagine(p, *args, taps=t), argnums=(0, 1),
                                                         has_aux=True)(carry["pi"], taps0)  # fmt: skip
        return loss, aux, {"enc": g_pi["enc"], "dec": fn.deferred_grads(aux["dec_in"], g_taps)}

    def policy_update(self, consts, carry, key, lr, pi_lr, batch=None, critic_out=False):
        """XMimic.policy_update. ``critic_out``: skip the critic phase and return its inputs instead, for a critic that runs
        on another GPU (AWL / AWLP with critic_offload) -> (carry, stats, {feat, F, r, d, k, lr}), the critic inputs being
        X = concat(feat, motion.goal(F))."""
        c, motion, H = self.cfg, consts["motion"], self.cfg.horizon
        k_s, k_n, k_i, k_c = jax.random.split(key, 4)
        start, o_hist0 = batch if batch is not None else self.sample_policy_batch(consts, carry["buf"], k_s)
        keys = jax.random.split(k_i, H)
        loss, aux, grads = self.policy_grads(carry, motion, start, o_hist0, keys)
        grads, gn = optim.clip_by_global_norm(grads, 1.0)
        if c.pi_clip_value > 0:
            grads = jax.tree.map(lambda g: jnp.clip(g, -c.pi_clip_value, c.pi_clip_value), grads)
        finite = jnp.isfinite(gn)
        grads = jax.tree.map(lambda g: jnp.where(finite, g, 0.0), grads)
        pi, opt_pi = fn.optim_step(carry["pi"], grads, carry["opt_pi"], self.pi_mask, pi_lr, betas=(c.pi_beta1, 0.95), ok=finite)
        avg = optim.ema(carry["avg"], (pi, carry["pi_b"]), c.pi_ema, carry["avg_n"] == 0)
        rews = sg(aux["r"])
        stats = {"pi": loss, "ir": ((aux["c_t"] * rews).sum(0) / H / 0.1).mean(), "gn": gn, "surv": aux["surv"].mean()}
        new = {"pi": pi, "opt_pi": opt_pi, "avg": avg, "avg_n": carry["avg_n"] + 1, "nonfinite": carry["nonfinite"] + (~finite).astype(jnp.int32)}
        if critic_out:
            crit = {"feat": self.critic_features(motion, aux), "F": aux["F_all"], "r": rews, "d": sg(aux["d"]), "k": k_c, "lr": lr}
            return {**carry, **new}, stats, crit
        X = jnp.concatenate((self.critic_features(motion, aux), aux["G_all"]), -1)
        critic, critic_b, opt_v, v_losses = self.critic_fit(carry, X, rews, sg(aux["d"]), k_c, lr)
        stats = {"pi": loss, "v": v_losses[-1], **{k: stats[k] for k in ("ir", "gn", "surv")}}
        return {**carry, **new, "critic": critic, "critic_b": critic_b, "opt_v": opt_v}, stats

    # ------------------------------------------------------------------ world model
    def sample_wm_batch(self, buf, key):
        """rb.sample_windows plus the mirroring of the reference wm_update (exact)."""
        c, mir, n = self.cfg, self.mirror, self.n_mirror
        w = rb.sample_windows(buf, key, c.wm_batch, c.wm_window)
        if n > 0:
            ms = lambda x: flatten_state(mir.state(unflatten_state(x)))  # noqa: E731
            for k_, f_ in (("s", ms), ("s1", ms), ("a", mir.joints), ("a_prev", mir.joints), ("xi", mir.xi)):
                w[k_] = jnp.concatenate((f_(w[k_][:n]), w[k_][n:]))
        return w

    def _wm_targets(self, w, wm_b):
        """The world-model loss inputs that do not depend on its parameters (start state, targets, frames, error scale)."""
        S0 = unflatten_state(w["s"][:, 0])
        T = unflatten_state(w["s1"])
        p, R, v, wv, q, qd = T
        return {"S0": S0, "xi_n": self.layout.normalize(w["xi"]), "T": T, "Rt": tr3(R), "Ht": tr3(yaw_mat(R)),
                "std": jnp.maximum(nets.rn_std(wm_b["inc_norm"], 1e-4), self.err_floor)}  # fmt: skip

    def wm_loss(self, wm, wm_b, w, pre, taps):
        """XMimic.wm_loss from precomputed targets, with bf16 taps for deferred weight gradients; also returns the stacked
        (window, B, in) bf16 inputs of every layer."""
        c = self.cfg
        S, xi_n = pre["S0"], pre["xi_n"]
        preds, ins_all = [], []
        for j in range(c.wm_window):
            S, ins = self._wm_step(wm, wm_b, S, w["a"][:, j], xi_n[:, j], taps=[t[j] for t in taps], defer=True)
            preds.append(S)
            ins_all.append(ins)
        ph, Rh, vh, wh, qh, qdh = tuple(jnp.stack([P[i] for P in preds], 1) for i in range(6))
        p, R, v, wv, q, qd = pre["T"]
        Rt = pre["Rt"]
        err = jnp.concatenate((mv3(pre["Ht"], ph - p), fn.so3_log(mm3(Rt, Rh)), mv3(Rt, vh - v), mv3(Rt, wh - wv), qh - q, qdh - qd), -1)
        e = ((err / pre["std"]) ** 2).mean(-1)
        loss = (e * w["mask"]).sum() / w["s"].shape[0]
        return loss, (e[:, -1], [jnp.stack([ins[i] for ins in ins_all]) for i in range(len(ins_all[0]))])

    def wm_update_presampled(self, carry, w, pre, lr):
        """XMimic.wm_update on presampled windows ``w`` and their targets ``pre``, with deferred weight gradients."""
        c = self.cfg
        wm, wm_b = carry["wm"], carry["wm_b"]
        taps0 = [jnp.zeros((c.wm_window, w["s"].shape[0], layer["w"].shape[1]), BF16) for layer in wm["net"]]
        (loss, (e_last, ins)), (g, g_taps) = jax.value_and_grad(lambda p, t: self.wm_loss(p, wm_b, w, pre, t),
                                                                argnums=(0, 1), has_aux=True)(wm, taps0)  # fmt: skip
        dg = fn.deferred_grads(ins, g_taps)
        grads = {"net": [{**gl, "w": d["w"], "b": d["b"]} for gl, d in zip(g["net"], dg)]}
        grads, _ = optim.clip_by_global_norm(grads, 1.0)
        wm, opt = fn.optim_step(wm, grads, carry["opt_wm"], self.wm_mask, lr, betas=(0.9, 0.999))
        m = w["mask"][:, -1]
        return {**carry, "wm": wm, "opt_wm": opt}, {"wm": loss / c.wm_window, "wm8": (e_last * m).sum() / jnp.maximum(m.sum(), 1)}

    def wm_updates(self, carry, k_w, lr):
        """The iteration's wm_batches world-model updates; all windows sampled up front (keys as in the reference)."""
        keys = jax.random.split(k_w, self.cfg.wm_batches)
        W = keyed_map(lambda k: self.sample_wm_batch(carry["buf"], k), keys)
        PRE = jax.vmap(lambda w: self._wm_targets(w, carry["wm_b"]))(W)
        return jax.lax.scan(lambda cr, wp: self.wm_update_presampled(cr, wp[0], wp[1], lr), carry, (W, PRE))
