"""Host side of a run: the train log, checkpoints written from a background thread, the pipelined dispatch loop and the
garbage-collection freeze. Nothing here touches the numerics; it keeps the GPUs from waiting for the host."""

from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import queue
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import nets


def freeze_gc():
    """Call right before the training clock starts: everything allocated during setup (modules, traced and compiled
    programs) moves to the permanent generation, so a full collection only scans the few objects the training loop keeps
    alive. Otherwise a generation-2 collection every ~500 iterations holds the GIL long enough to idle the GPUs of the
    multi-GPU layouts (AWLP dropped from 36 to 27 it/s in those log intervals)."""
    gc.collect()
    gc.freeze()


def compile_all(jobs):
    """Run compile jobs (zero-argument callables, e.g. ``lowered.compile``) in parallel threads and return their results in
    order. XLA's backend compilation releases the GIL, so independent programs compile side by side; lowering holds the GIL
    and stays serial (it happens before). XMIMIC_COMPILE_WORKERS sets the number of threads (default 8; 1 = one by one)."""
    workers = int(os.environ.get("XMIMIC_COMPILE_WORKERS", "8"))
    with concurrent.futures.ThreadPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
        futures = [pool.submit(job) for job in jobs]
        return [f.result() for f in futures]


def make_unflat(template):
    """Inverse of algorithm.flat32 for trees shaped like ``template`` (numpy or jax vectors)."""
    leaves, tdef = jax.tree.flatten(template)
    spec = [(x.shape, int(np.prod(x.shape))) for x in leaves]

    def unflat(v):
        out, i = [], 0
        for shape, n in spec:
            out.append(v[i : i + n].reshape(shape))
            i += n
        return jax.tree.unflatten(tdef, out)

    return unflat


def export_policy(path, avg, meta_extra, test=None):
    """Save the averaged (deployed) policy and its normalizers as an npz. ``test`` = (observation history, goal, action
    computed here) lets tools/xmimic_to_blink.py check its conversion to BlinkMimic's format."""
    flat = {}
    if test is not None:
        flat["test_oh"], flat["test_g"], flat["test_a"] = (np.asarray(x, np.float32) for x in test)
    params, bufs = avg
    for name in ("enc", "dec"):
        for i, layer in enumerate(params[name]):
            flat[f"{name}.{i}.w"] = np.asarray(layer["w"], np.float32)
            flat[f"{name}.{i}.b"] = np.asarray(layer["b"], np.float32)
    for name in ("obs_norm", "goal_norm"):
        for k, v in bufs[name].items():
            flat[f"{name}.{k}"] = np.asarray(v, np.float32)
    np.savez(path + ".tmp.npz", **flat, meta=json.dumps(meta_extra))
    os.replace(path + ".tmp.npz", path)


class RunLog:
    """train_log.csv rows and console lines per log interval, and checkpoints from a background thread, so the host goes
    straight back to dispatching work.

    Timing: a dispatch that contains a data-dependent loop returns only near the end of its program (XLA:GPU reads loop
    predicates on the host), so the host cannot timestamp program k after dispatching program k+1. A watcher thread blocks
    on each program's outputs as soon as it is dispatched (``submitted``) and records when the device finished it."""

    COLS = ["iteration", "wall_min", "sim_steps", "wm", "wm8", "pi", "v", "ir", "gn", "surv", "sim_r", "ep_len", "ep_ret", "term_frac",
            "it_per_s", "lr", "pi_lr", "nonfinite"]  # fmt: skip

    def __init__(self, xm, cfg, t0, avg_template, args_meta, extra_cols=()):
        self.cfg, self.t0 = cfg, t0
        self.args_meta = {**args_meta, "robot": xm.meta.get("robot")}
        self.cols = self.COLS + list(extra_cols)
        self.f = open(os.path.join(cfg.log_dir, "train_log.csv"), "a")
        self.f.write(",".join(self.cols) + "\n")
        self.acc, self.t_log, self.it_log, self.decay_at = [], time.time(), 0, -1
        self.unflat = make_unflat(avg_template)
        self.pool = concurrent.futures.ThreadPoolExecutor(1)
        self.futures = []
        # the checkpoint's test vector (tools/xmimic_to_blink.py checks the converted policy against it), on the CPU
        self.cpu = jax.devices("cpu")[0]
        kt1, _ = jax.random.split(jax.random.PRNGKey(1234))
        oh = 0.5 * jax.random.normal(kt1, (4, cfg.hist, xm.obs_dim))
        g = xm.motion.goal(jnp.asarray([100, 5000, 100000, 300000], jnp.int32), cfg.goal_frames)
        self.test_in = jax.device_put((oh, g), self.cpu)
        self.test_act = jax.jit(lambda avg, oh, g: nets.pi_apply(avg[0], avg[1], oh, g, xm.ref))
        self.done, self.q = {}, queue.Queue()
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        while True:
            it, st, ev = self.q.get()
            jax.block_until_ready(st)
            self.done[it] = time.time()
            ev.set()

    def submitted(self, it, st):
        """Call right after dispatching the work whose stats ``st`` end at iteration ``it``."""
        ev = threading.Event()
        self.q.put((it, st, ev))
        self.done[("ev", it)] = ev

    def add(self, it, st, avg_flat, extra=None):
        """Stats of the work ending at iteration ``it`` (waits until it is done); returns the training minutes so far."""
        c = self.cfg
        st = jax.device_get(st)
        ev = self.done.pop(("ev", it), None)
        if ev is not None:
            ev.wait()
        now = self.done.pop(it, None) or time.time()
        self.acc.append(st)
        if it % c.log_every == 0 or it >= c.iterations:
            s = {k: np.concatenate([a[k] for a in self.acc]) for k in st}
            self.acc = []
            ips = (it - self.it_log) / (now - self.t_log)
            self.t_log, self.it_log = now, it
            nd = float(s["n_done"].sum())
            ep_len = float(s["ep_len_sum"].sum()) / max(nd, 1)
            ep_ret = float(s["ep_ret_norm_sum"].sum()) / max(nd, 1) if nd > 0 else float("nan")
            row = {"iteration": it, "wall_min": (now - self.t0) / 60, "sim_steps": float(s["sim_steps"][-1]),
                   "wm": float(s["wm"].mean()), "wm8": float(s["wm8"].mean()), "pi": float(s["pi"].mean()), "v": float(s["v"].mean()),
                   "ir": float(s["ir"].mean()), "gn": float(s["gn"].mean()), "surv": float(s["surv"].mean()),
                   "sim_r": float(s["r"].sum() / max(s["n_coll"].sum(), 1)), "ep_len": ep_len, "ep_ret": ep_ret,
                   "term_frac": float(s["term_sum"].sum()) / max(nd, 1), "it_per_s": ips, "lr": float(s["lr"][-1]),
                   "pi_lr": float(s["pi_lr"][-1]), "nonfinite": int(s["nonfinite"][-1]), **(extra or {})}  # fmt: skip
            self.f.write(",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in self.cols) + "\n")
            self.f.flush()
            probe = (f" | probe fails {int(s['ep_n_fail'][-1])}" if "ep_n_fail" in s else "") + "".join(
                f" | {k} {v}" for k, v in (extra or {}).items())
            print(f"it {it} | {row['wall_min']:.2f} min | wm {row['wm']:.4f} e8 {row['wm8']:.4f} | pi {row['pi']:.3f} v {row['v']:.4f} "
                  f"imag_r {row['ir']:.3f} surv {row['surv']:.3f} | sim_r {row['sim_r']:.3f} ep_len {ep_len:.1f} ep_ret {ep_ret:.3f} "
                  f"term {row['term_frac']:.2f} | {ips:.2f} it/s | pi_lr {row['pi_lr']:.2e}{probe}", flush=True)  # fmt: skip
            if "n_mismatch" in s and s["n_mismatch"].sum() > 0:
                print(f"WARNING: the host's collection counts disagree with the device schedule in {int(s['n_mismatch'].sum())} "
                      f"iterations before it {it}", flush=True)
            da = int(s["decay_at"].max())
            if self.decay_at < 0 <= da:
                self.decay_at = da
                print(f"lr decay trigger at it {da} (ep_ret {ep_ret:.3f})", flush=True)
        if it % c.ckpt_every == 0 or it >= c.iterations:
            self.futures.append(self.pool.submit(self._ckpt, it, now - self.t0, float(st["sim_steps"][-1]), avg_flat))
        return (now - self.t0) / 60

    def _ckpt(self, it, wall, sim_steps, avg_flat):
        avg = self.unflat(np.asarray(jax.device_get(avg_flat)))
        test_a = self.test_act(jax.device_put(avg, self.cpu), *self.test_in)
        export_policy(os.path.join(self.cfg.log_dir, f"ckpt_{it:06d}.npz"), avg,
                      {"iteration": it, "wall_time": wall, "sim_steps": sim_steps, "args": self.args_meta}, (*self.test_in, test_a))

    def close(self):
        for fu in self.futures:
            fu.result()  # re-raises a failed checkpoint write
        self.pool.shutdown()
        self.f.close()


def run_pipelined(dispatch, log, iterations, max_minutes=0.0, on_done=None):
    """Dispatch program k+1, then log program k: the GPU never waits for the host. ``dispatch()`` -> (it, stats, avg_flat)
    of the program it just queued; ``on_done(it, avg_flat)`` runs once program k's results are in (FP: prober hand-offs)
    and returns extra log columns. Returns the number of iterations dispatched."""
    it, prev, stop = 0, None, False
    while True:
        cur = None
        if it < iterations and not stop:
            cur = dispatch()
            it = cur[0]
            log.submitted(it, cur[1])
        if prev is not None:
            extra = on_done(prev[0], prev[2]) if on_done else None
            if log.add(*prev, extra=extra) >= max_minutes > 0 and not stop:
                print(f"time budget {max_minutes} min reached at it {prev[0]}", flush=True)
                stop = True
        if cur is None:
            break
        prev = cur
    log.close()
    return it


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
