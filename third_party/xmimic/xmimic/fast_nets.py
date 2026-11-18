"""Building blocks of the fast learner (xmimic/fast.py), each documented against its reference in nets.py, rot.py or
optim.py:

- exact: the same operations on the same values (only hoisted, batched or reordered across independent data);
- fp32:  the same math with a different fp32 rounding order (a hand-written VJP, a 4-term sum instead of a reduction);
- bf16:  the same math with bf16 roundings in other places (deferred weight gradients, bf16 GEMM outputs).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

BF16 = jnp.bfloat16
F32 = jnp.float32
sg = jax.lax.stop_gradient


# ---------------------------------------------------------------- dense layers
def mm(xb, wb):
    """bf16 x bf16 GEMM with fp32 accumulation and a bf16 output, upcast to fp32 for the bias and LayerNorm (torch
    autocast's bf16 Linear). JAX then transposes it with bf16 cotangents: every backward GEMM runs on bf16 tensor cores.
    (bf16; the reference nets.mlp_apply keeps the fp32 output of the accumulator.)"""
    return jnp.dot(xb, wb, preferred_element_type=BF16).astype(F32)


def dense(x, layer, *, defer=False, tap=None):
    """One layer before its activation: bf16(x) @ bf16(w) + b.

    defer: the weight and bias are stop-gradient'ed and ``tap`` (zeros) is added to the pre-activation; the cotangent of the
    tap is the pre-activation cotangent dy, from which ``deferred_grads`` forms dW = sum x^T dy and db = sum dy once over
    all steps. Returns (y, the bf16 input)."""
    xb = x.astype(BF16)
    w, b = layer["w"], layer["b"]
    if defer:
        w, b = sg(w), sg(b)
    y = mm(xb, w.astype(BF16)) + b
    if tap is not None:
        y = y + tap.astype(F32)
    return y, xb


def deferred_grads(xs, dys):
    """The weight and bias gradients of a layer applied at many steps, as one GEMM per layer: sum over all leading axes of
    x^T dy, and of dy. The reference's per-step transposes accumulate them one step at a time inside the sequential
    backward chain. (bf16: dy arrives bf16-rounded through the bf16 taps; one GEMM on bf16 operands, fp32 accumulation.)"""
    out = []
    for x, dy in zip(xs, dys):
        xr = x.reshape(-1, x.shape[-1]).astype(BF16)
        dr = dy.reshape(-1, dy.shape[-1])
        dw = jax.lax.dot_general(xr, dr.astype(BF16), (((0,), (0,)), ((), ())), preferred_element_type=F32)
        out.append({"w": dw, "b": dr.astype(F32).sum(0)})
    return out


# ---------------------------------------------------------------- LayerNorm with a hand-written backward pass
@partial(jax.custom_vjp, nondiff_argnums=(3,))
def layer_norm(h, g, b, eps=1e-5):
    """nets.mlp_apply's two-pass LayerNorm (forward exact). The backward pass computes its two row reductions side by side
    instead of the dependent chain autodiff derives from the two-pass forward (fp32)."""
    return _ln_fwd(h, g, b, eps)[0]


def _ln_fwd(h, g, b, eps):
    mu = h.mean(-1, keepdims=True)
    var = ((h - mu) ** 2).mean(-1, keepdims=True)
    r = jax.lax.rsqrt(var + eps)
    xh = (h - mu) * r
    return xh * g + b, (xh, r, g)


def _ln_bwd(eps, res, dy):
    # dh = r (dxh - mean(dxh) - xh mean(dxh xh)): two sibling row reductions (one multi-output kernel)
    xh, r, g = res
    dxh = dy * g
    m1 = dxh.mean(-1, keepdims=True)
    m2 = (dxh * xh).mean(-1, keepdims=True)
    dh = r * (dxh - m1 - xh * m2)
    red = tuple(range(dy.ndim - 1))
    return dh, (dy * xh).sum(red), dy.sum(red)


layer_norm.defvjp(lambda h, g, b, eps: _ln_fwd(h, g, b, eps), _ln_bwd)


def mlp_apply(layers, x, act, *, defer=False, taps=None, out_dtype=F32):
    """nets.mlp_apply with bf16 GEMM outputs and the LayerNorm above. Returns (output, [bf16 input of every layer]) for
    deferred weight gradients."""
    h, ins = x, []
    for i, layer in enumerate(layers):
        h, xb = dense(h, layer, defer=defer, tap=None if taps is None else taps[i])
        ins.append(xb)
        if i == len(layers) - 1:
            return h.astype(out_dtype), ins
        if "ln_g" in layer:
            h = layer_norm(h, layer["ln_g"], layer["ln_b"])
        h = act(h)


# ---------------------------------------------------------------- rotations without 4-element reductions, argmax or gathers
def _norm4(q):
    return jnp.sqrt(q[..., 0:1] * q[..., 0:1] + q[..., 1:2] * q[..., 1:2] + q[..., 2:3] * q[..., 2:3] + q[..., 3:4] * q[..., 3:4])


def mat_to_quat(m):
    """rot.mat_to_quat with the largest-diagonal branch chosen by comparisons (argmax's first-max tie rule) and selected
    with where: the same values and the same gradient (the reference's take_along_axis transposes to a scatter-add), but
    fully elementwise, so it fuses. The final 4-element norm is written out (fp32)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = jnp.sqrt(jnp.maximum(1 + m00 + m11 + m22, 1e-10)) / 2
    qx = jnp.sqrt(jnp.maximum(1 + m00 - m11 - m22, 1e-10)) / 2
    qy = jnp.sqrt(jnp.maximum(1 - m00 + m11 - m22, 1e-10)) / 2
    qz = jnp.sqrt(jnp.maximum(1 - m00 - m11 + m22, 1e-10)) / 2
    c0 = jnp.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1)
    c1 = jnp.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1)
    c2 = jnp.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1)
    c3 = jnp.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1)
    s0 = ((qw >= qx) & (qw >= qy) & (qw >= qz))[..., None]
    s1 = ((qx >= qy) & (qx >= qz))[..., None]
    s2 = (qy >= qz)[..., None]
    q = jnp.where(s0, c0, jnp.where(s1, c1, jnp.where(s2, c2, c3)))
    q = q / jnp.maximum(_norm4(q), 1e-12)
    return q * jnp.where(q[..., :1] < 0, -1.0, 1.0)


def quat_to_rotvec(q):
    """rot.quat_to_rotvec (exact)."""
    q = q * jnp.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = jnp.linalg.norm(v, axis=-1, keepdims=True)
    half = jnp.arctan2(sin_half, q[..., :1])
    scale = jnp.where(sin_half < 1e-6, 2.0 / jnp.maximum(q[..., :1], 1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def so3_log(R):
    return quat_to_rotvec(mat_to_quat(R))


# ---------------------------------------------------------------- Muon with same-shape matrices stacked
NS_COEFFS = (3.4445, -4.775, 2.0315)


def newton_schulz_batched(G, steps: int = 5, eps: float = 1e-7):
    """optim.newton_schulz on a stack (n, rows, cols) of same-shape matrices: one batched GEMM per product instead of n.
    Every matrix sees the reference arithmetic (bf16 operands, fp32 accumulation, bf16 rounding)."""
    a, b, c = NS_COEFFS
    X = G.astype(BF16)
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = jnp.swapaxes(X, -1, -2)
    nrm = jnp.sqrt(jnp.sum(jnp.square(X.astype(F32)), axis=(-2, -1), keepdims=True))
    X = X / jnp.maximum(nrm, eps).astype(BF16)
    bmm = partial(jnp.einsum, preferred_element_type=F32)
    for _ in range(steps):
        A = bmm("nij,nkj->nik", X, X).astype(BF16)
        B = (b * A.astype(F32) + c * bmm("nij,njk->nik", A, A)).astype(BF16)
        X = (a * X.astype(F32) + bmm("nij,njk->nik", B, X)).astype(BF16)
    return jnp.swapaxes(X, -1, -2) if tall else X


def optim_step(params, grads, state, mask, lr, betas=(0.9, 0.999), eps: float = 1e-8, momentum: float = 0.95, ok=None):
    """optim.step with the Muon matrices of equal shape orthogonalized together (bf16: a batched GEMM may accumulate in
    another order, and the bf16 Newton-Schulz iteration turns that into 1-ulp bf16 differences of the update)."""
    b1, b2 = betas
    t = state["t"] + 1
    bc1, bc2 = 1 - b1 ** t.astype(F32), 1 - b2 ** t.astype(F32)
    flat_p, tdef = jax.tree.flatten(params)
    flat_g = tdef.flatten_up_to(grads)
    flat_s = tdef.flatten_up_to(state["leaves"])
    flat_m = tdef.flatten_up_to(mask)
    new_p, new_s = [None] * len(flat_p), [None] * len(flat_p)
    groups = {}
    for i, (p, g, s, m) in enumerate(zip(flat_p, flat_g, flat_s, flat_m)):
        if m:
            groups.setdefault(p.shape, []).append(i)
            continue
        mm_ = b1 * s["m"] + (1 - b1) * g
        vv = b2 * s["v"] + (1 - b2) * g * g
        new_p[i] = p - (lr / bc1) * mm_ / (jnp.sqrt(vv) / jnp.sqrt(bc2) + eps)
        new_s[i] = {"m": mm_, "v": vv}
    for shape, idx in groups.items():
        bufs = [momentum * flat_s[i]["buf"] + (1 - momentum) * flat_g[i] for i in idx]
        nest = jnp.stack([flat_g[i] + momentum * (bf - flat_g[i]) for i, bf in zip(idx, bufs)])
        upd = newton_schulz_batched(nest).astype(F32)
        ratio = 0.2 * max(shape[0], shape[1]) ** 0.5
        for k, i in enumerate(idx):
            new_p[i] = flat_p[i] - lr * ratio * upd[k]
            new_s[i] = {"buf": bufs[k]}
    new_p = tdef.unflatten(new_p)
    new_s = {"leaves": tdef.unflatten(new_s), "t": t}
    if ok is None:
        return new_p, new_s
    keep = lambda new, old: jax.tree.map(lambda a, b: jnp.where(ok, a, b), new, old)  # noqa: E731
    return keep(new_p, params), keep(new_s, state)
