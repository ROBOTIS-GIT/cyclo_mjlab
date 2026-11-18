"""MuonAdam and Adam as pure functions, matching blinkmimic/optim.py (torch.optim.Muon with match_rms_adamw, torch Adam).

Muon runs on the leaves where ``mask`` is True (hidden weight matrices), Adam on the rest. ``ok`` (0-dim bool) gates the whole
step: where it is False, parameters and every optimizer state are returned unchanged (BlinkMimic's --no_sync skip of a
non-finite policy update).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

NS_COEFFS = (3.4445, -4.775, 2.0315)


def newton_schulz(G, steps: int = 5, eps: float = 1e-7):
    """torch.optim._muon._zeropower_via_newtonschulz: quintic iteration in bf16 on the wide orientation of G."""
    a, b, c = NS_COEFFS
    X = G.astype(jnp.bfloat16)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    X = X / jnp.maximum(jnp.linalg.norm(X.astype(jnp.float32)), eps).astype(jnp.bfloat16)
    for _ in range(steps):
        A = jnp.dot(X, X.T, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        B = (b * A.astype(jnp.float32) + c * jnp.dot(A, A, preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
        X = (a * X.astype(jnp.float32) + jnp.dot(B, X, preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
    return X.T if tall else X


def init(params, mask):
    def leaf(p, m):
        return {"buf": jnp.zeros_like(p)} if m else {"m": jnp.zeros_like(p), "v": jnp.zeros_like(p)}

    return {"leaves": jax.tree.map(leaf, params, mask), "t": jnp.zeros((), jnp.int32)}


def clip_by_global_norm(grads, max_norm: float):
    """torch.nn.utils.clip_grad_norm_: returns the clipped grads and the total norm before clipping."""
    norm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree.leaves(grads)))
    coef = jnp.minimum(max_norm / (norm + 1e-6), 1.0)
    return jax.tree.map(lambda g: g * coef, grads), norm


def step(params, grads, state, mask, lr, betas=(0.9, 0.999), eps: float = 1e-8, momentum: float = 0.95, ok=None):
    """One MuonAdam step (Adam only where mask is all False). lr may be a traced scalar."""
    b1, b2 = betas
    t = state["t"] + 1
    bc1, bc2 = 1 - b1**t.astype(jnp.float32), 1 - b2**t.astype(jnp.float32)

    def leaf(p, g, s, m):
        if m:
            buf = momentum * s["buf"] + (1 - momentum) * g
            upd = newton_schulz(g + momentum * (buf - g)).astype(jnp.float32)
            ratio = 0.2 * max(p.shape[0], p.shape[1]) ** 0.5
            return p - lr * ratio * upd, {"buf": buf}
        mm = b1 * s["m"] + (1 - b1) * g
        vv = b2 * s["v"] + (1 - b2) * g * g
        return p - (lr / bc1) * mm / (jnp.sqrt(vv) / jnp.sqrt(bc2) + eps), {"m": mm, "v": vv}

    flat_p, tdef = jax.tree.flatten(params)
    flat_g = tdef.flatten_up_to(grads)
    flat_s = tdef.flatten_up_to(state["leaves"])
    flat_m = tdef.flatten_up_to(mask)
    out = [leaf(p, g, s, m) for p, g, s, m in zip(flat_p, flat_g, flat_s, flat_m)]
    new_p = tdef.unflatten([o[0] for o in out])
    new_s = {"leaves": tdef.unflatten([o[1] for o in out]), "t": t}
    if ok is None:
        return new_p, new_s
    keep = lambda new, old: jax.tree.map(lambda a, b: jnp.where(ok, a, b), new, old)  # noqa: E731
    return keep(new_p, params), keep(new_s, state)


def ema(avg, live, decay: float, first):
    """AveragedModel EMA (get_ema_multi_avg_fn): the first update copies, later ones lerp toward the live weights."""
    return jax.tree.map(lambda a, l: jnp.where(first, l, a + (1 - decay) * (l - a)), avg, live)
