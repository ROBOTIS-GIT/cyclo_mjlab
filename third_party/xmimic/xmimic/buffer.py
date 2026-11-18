"""Fixed-size replay buffer on the device, laid out (time, world): a port of blinkmimic/buffer.py in its --graph_early form
(fill level and write head live on the device; uniform draws are scaled by the fill level, so nothing syncs with the host).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

FIELDS = ("s", "s1", "a", "a_prev", "xi", "f", "ep_step", "cut")


def init(T: int, N: int, sdim: int, J: int, xi_dim: int):
    """An empty buffer of T steps for N worlds: states before and after, action, previous action, sim context, reference
    frame, episode step and the episode-cut flag of every transition."""
    z = lambda *shape, dt=jnp.float32: jnp.zeros((T, N) + shape, dt)  # noqa: E731
    return {"s": z(sdim), "s1": z(sdim), "a": z(J), "a_prev": z(J), "xi": z(xi_dim), "f": z(dt=jnp.int32),
            "ep_step": z(dt=jnp.int32), "cut": z(dt=bool), "ptr": jnp.zeros((), jnp.int32), "size": jnp.zeros((), jnp.int32)}  # fmt: skip


def add(buf, tr):
    """Write one step of transitions (all worlds) at the write head."""
    T = buf["s"].shape[0]
    ptr = buf["ptr"]
    out = {k: jax.lax.dynamic_update_index_in_dim(buf[k], tr[k].astype(buf[k].dtype), ptr, 0) for k in FIELDS}
    return {**out, "ptr": (ptr + 1) % T, "size": jnp.minimum(buf["size"] + 1, T)}


def _row(buf, age):
    """Buffer row of the transition ``age`` steps after the oldest one."""
    T = buf["s"].shape[0]
    return (buf["ptr"] - buf["size"] + age) % T


def _uniform_int(key, n, B):
    """B integers uniform in [0, n) for a traced n."""
    return jnp.minimum((jax.random.uniform(key, (B,)) * n).astype(jnp.int32), n - 1)


def sample_windows(buf, key, B: int, k: int):
    """B windows of k consecutive steps (B, k, ...) for the world-model loss, and their mask: step j counts while no
    episode cut happened in steps 0..j-1."""
    kt, ke = jax.random.split(key)
    N = buf["s"].shape[1]
    t0 = _uniform_int(kt, buf["size"] - k + 1, B)
    env = jax.random.randint(ke, (B,), 0, N)
    rows = _row(buf, t0[:, None] + jnp.arange(k))
    e = env[:, None]
    out = {name: buf[name][rows, e] for name in ("s", "s1", "a", "a_prev", "xi", "cut")}
    d = out["cut"].astype(jnp.float32)
    out["mask"] = jnp.cumprod(jnp.concatenate((jnp.ones((B, 1)), 1 - d[:, :-1]), -1), -1)
    return out


def sample_starts(buf, key, B: int, hist: int):
    """B imagination start points and the rows of their observation history (oldest first). The history stays inside the
    start's episode and inside the buffer (earlier entries repeat the episode's first step)."""
    ka, ke = jax.random.split(key)
    N = buf["s"].shape[1]
    age = _uniform_int(ka, buf["size"], B)
    env = jax.random.randint(ke, (B,), 0, N)
    row = _row(buf, age)
    ep_step = buf["ep_step"][row, env]
    back = jnp.arange(hist - 1, -1, -1)
    back = jnp.minimum(back[None, :], ep_step[:, None])
    back = jnp.minimum(back, age[:, None])
    hrows = _row(buf, age[:, None] - back)
    henv = env[:, None]
    start = {k: buf[k][row, env] for k in ("s", "a_prev", "xi", "f")}
    hist_d = {k: buf[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
    return start, hist_d
