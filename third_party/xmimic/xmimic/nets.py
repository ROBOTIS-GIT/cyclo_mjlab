"""World model, policy (motion encoder + shared decoder) and critic as pure functions on parameter pytrees, a port of
blinkmimic/networks.py.

Matmuls take bf16 inputs and accumulate in fp32 (BlinkMimic's --amp bf16); LayerNorm, normalizers, the state integration
and everything outside the MLPs stay fp32. Running normalizers are separate "buffer" pytrees: they are updated from data,
never by gradients, and are part of the policy's weight average (torch AveragedModel with use_buffers=True).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .mdp import invariant_features
from .rot import mm3, mv3, so3_exp, so3_log, tr3, yaw_mat

BF16 = jnp.bfloat16
# Matmul output dtype. The trainer sets bfloat16, as torch autocast does: the cotangent that reaches each matmul is then
# bf16 too, and JAX's transpose rule keeps the backward GEMMs on bf16 tensor cores (with float32 outputs every backward GEMM
# becomes float32 x float32, TF32 at best). The evaluator keeps the float32 default.
MM_OUT = jnp.float32


# ---------------------------------------------------------------- MLPs
def mlp_init(key, sizes: list[int], layer_norm: bool = False):
    """torch.nn.Linear's default init (U(+-1/sqrt(fan_in)) for weights and biases); weights stored (in, out)."""
    layers = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (i, o) in zip(keys, zip(sizes[:-1], sizes[1:])):
        kw, kb = jax.random.split(k)
        bound = 1.0 / np.sqrt(i)
        layer = {"w": jax.random.uniform(kw, (i, o), minval=-bound, maxval=bound), "b": jax.random.uniform(kb, (o,), minval=-bound, maxval=bound)}
        layers.append(layer)
    if layer_norm:
        for layer in layers[:-1]:
            n = layer["b"].shape[0]
            layer["ln_g"], layer["ln_b"] = jnp.ones(n), jnp.zeros(n)
    return layers


def mlp_apply(layers, x, act, out_dtype=jnp.float32):
    """Linear (bf16 matmul) -> [LayerNorm] -> activation for every layer but the last."""
    h = x
    for i, layer in enumerate(layers):
        h = jnp.dot(h.astype(BF16), layer["w"].astype(BF16), preferred_element_type=MM_OUT).astype(jnp.float32) + layer["b"]
        if i == len(layers) - 1:
            return h.astype(out_dtype)
        if "ln_g" in layer:
            mu = h.mean(-1, keepdims=True)
            var = ((h - mu) ** 2).mean(-1, keepdims=True)
            h = (h - mu) * jax.lax.rsqrt(var + 1e-5) * layer["ln_g"] + layer["ln_b"]
        h = act(h)


def muon_mask(layers):
    """True for the hidden weight matrices (every Linear except the first and last), as blinkmimic.optim.MuonAdam."""
    return [{k: (k == "w" and 0 < i < len(layers) - 1) for k in layer} for i, layer in enumerate(layers)]


# ---------------------------------------------------------------- running normalizer
def rn_init(dim: int):
    return {"mean": jnp.zeros(dim), "var": jnp.ones(dim), "count": jnp.zeros(())}


def rn_update(st, x, max_count: float = 2e8):
    """blinkmimic RunningNorm.update (no_sync form): merge the batch statistics until max_count samples, then freeze."""
    x = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
    n = x.shape[0]
    m, v = x.mean(0), x.var(0)
    cnt = st["count"]
    tot = cnt + n
    delta = m - st["mean"]
    mean = st["mean"] + delta * n / tot
    var = (st["var"] * cnt + v * n + delta**2 * cnt * n / tot) / tot
    live = cnt < max_count
    return {"mean": jnp.where(live, mean, st["mean"]), "var": jnp.where(live, var, st["var"]), "count": jnp.where(live, tot, cnt)}


def rn_apply(st, x, eps: float = 1e-2):
    return (x - st["mean"]) / jnp.sqrt(st["var"] + eps**2)


def rn_std(st, eps: float = 1e-2):
    return jnp.sqrt(st["var"] + eps**2)


# ---------------------------------------------------------------- world model (semi-implicit, BlinkMimic --wm_structured)
# The world model predicts normalized state increments in the root (heading) frame. Positions are integrated with a
# semi-implicit drift from the old and new velocities; the network predicts only the residual.
DRIFT_W = 0.625  # (n + 1) / (2 n) for n = 4 semi-implicit substeps


def wm_init(key, in_dim: int, out_dim: int, hidden=(1024, 1024, 1024)):
    layers = mlp_init(key, [in_dim, *hidden, out_dim], layer_norm=True)
    layers[-1]["w"] = jnp.zeros_like(layers[-1]["w"])
    layers[-1]["b"] = jnp.zeros_like(layers[-1]["b"])
    return {"net": layers}, {"in_norm": rn_init(in_dim), "out_norm": rn_init(out_dim), "inc_norm": rn_init(out_dim)}


def wm_inputs(S, a, xi_n):
    return jnp.concatenate((invariant_features(S), a, xi_n), -1)


def _drift(S, v1, w1, qd1, dt):
    p, R, v, w, q, qd = S
    a, b = dt * (1 - DRIFT_W), dt * DRIFT_W
    return mv3(tr3(yaw_mat(R)), a * v + b * v1), mv3(tr3(R), a * w + b * w1), a * qd + b * qd1


def increments(S, S1):
    p, R, v, w, q, qd = S
    p1, R1, v1, w1, q1, qd1 = S1
    Rt = tr3(R)
    return jnp.concatenate((mv3(tr3(yaw_mat(R)), p1 - p), so3_log(mm3(Rt, R1)), mv3(Rt, v1 - v), mv3(Rt, w1 - w), q1 - q, qd1 - qd), -1)


def wm_targets(S, S1, dt):
    inc = increments(S, S1)
    J = S[4].shape[-1]
    dp, dth, dv, dw, dq, dqd = jnp.split(inc, np.cumsum([3, 3, 3, 3, J])[:5], -1)
    gp, gth, gq = _drift(S, S1[2], S1[3], S1[5], dt)
    return jnp.concatenate((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)


def wm_integrate(S, d, dt):
    p, R, v, w, q, qd = S
    J = q.shape[-1]
    dp, dth, dv, dw, dq, dqd = jnp.split(d, np.cumsum([3, 3, 3, 3, J])[:5], -1)
    v1, w1, qd1 = v + mv3(R, dv), w + mv3(R, dw), qd + dqd
    gp, gth, gq = _drift(S, v1, w1, qd1, dt)
    return p + mv3(yaw_mat(R), dp + gp), mm3(R, so3_exp(dth + gth)), v1, w1, q + dq + gq, qd1


def wm_step(params, bufs, S, a, xi_n, dt):
    """One imagined control step: state, action and normalized sim context -> next state."""
    y = mlp_apply(params["net"], rn_apply(bufs["in_norm"], wm_inputs(S, a, xi_n)), jax.nn.silu)
    return wm_integrate(S, y * rn_std(bufs["out_norm"], 1e-4) + bufs["out_norm"]["mean"], dt)


def state_error(S_hat, S):
    """Prediction error in the true state's root frame (the world-model loss before normalization)."""
    p, R, v, w, q, qd = S
    ph, Rh, vh, wh, qh, qdh = S_hat
    Rt = tr3(R)
    return jnp.concatenate((mv3(tr3(yaw_mat(R)), ph - p), so3_log(mm3(Rt, Rh)), mv3(Rt, vh - v), mv3(Rt, wh - w), qh - q, qdh - qd), -1)


# ---------------------------------------------------------------- policy
def pi_init(key, obs_dim: int, goal_dim: int, act_dim: int, hist: int, z_dim: int = 32,
            enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256)):  # fmt: skip
    ke, kd = jax.random.split(key)
    enc = mlp_init(ke, [goal_dim, *enc_hidden, z_dim])
    dec = mlp_init(kd, [z_dim + obs_dim * hist, *dec_hidden, act_dim])
    dec[-1]["w"] = dec[-1]["w"] * 0.01
    dec[-1]["b"] = jnp.zeros_like(dec[-1]["b"])
    return {"enc": enc, "dec": dec}, {"obs_norm": rn_init(obs_dim), "goal_norm": rn_init(goal_dim)}


def pi_encode(params, bufs, goal):
    return jnp.tanh(mlp_apply(params["enc"], rn_apply(bufs["goal_norm"], goal), jax.nn.elu))


def pi_apply(params, bufs, obs_hist, goal, ref, z=None):
    """a = dec(enc(g), o_{t-9..t}) + (q_ref(f+1) - q0) / alpha (--pi_ref_skip); ref = (goal offset of q(f+1), q0, alpha)."""
    off, q0, alpha = ref
    o = rn_apply(bufs["obs_norm"], obs_hist).reshape(obs_hist.shape[:-2] + (-1,))
    if z is None:
        z = pi_encode(params, bufs, goal)
    a = mlp_apply(params["dec"], jnp.concatenate((z, o), -1), jax.nn.elu)
    J = a.shape[-1]
    return a + (goal[..., off : off + J] - q0) / alpha


# ---------------------------------------------------------------- critic
def critic_init(key, in_dim: int):
    return {"net": mlp_init(key, [in_dim, 256, 128, 1])}, {"norm": rn_init(in_dim)}


def critic_apply(params, bufs, x):
    return mlp_apply(params["net"], rn_apply(bufs["norm"], x), jax.nn.elu)[..., 0]
