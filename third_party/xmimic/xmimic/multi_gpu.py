"""XMimic on several GPUs: one host process dispatches a compiled program per GPU and iteration, and the programs run
concurrently. The layouts (--layout), by the roles of their GPUs:

  FP1   one GPU: the 1GPU trainer, probe sweeps in a second process on the same GPU (CUDA MPS)
  FP    the 1GPU trainer (25-iteration programs) | prober
  AL    actor | policy learner, probe sweeps in a second process on the learner's GPU (CUDA MPS)
  ALP   actor | policy learner | prober
  AWL   actor | world-model learner (+ critic) | policy learner, probe sweeps in a second process on the world-model GPU
  AWLP  actor | world-model learner (+ critic) | policy learner | prober

Actor and learners (AL, ALP, AWL, AWLP): in iteration i the actor collects with the policy of learner iteration i-1 and
samples the next imagination batch, while the policy learner updates on the batch and world model of iteration i-1. So
collection acts with a one-iteration-old policy and imagination uses a one-iteration-older world model: the staleness
InstantTrack v1.3 uses on one GPU, here across GPUs. AWL / AWLP move the world-model updates to their own GPU, which keeps
an exact replica of the actor's replay buffer; with critic_offload the critic fit runs there too, so the policy update
bootstraps with a critic one iteration older. Policy weights travel as bf16 (every matmul casts them to bf16 anyway).

Prober: BlinkMimic's evaluation-guided resets without stalling training. It sweeps the averaged policy over all clips under
evaluation conditions, one dispatched program per sweep, and hands the failure focus to the trainer when the sweep is done.
At 25-40 it/s a sweep (about 4 s) spans 100-150 iterations, so a sweep that falls due while one runs is skipped.

Dispatch: XLA:GPU reads the predicate of a data-dependent while loop on the host before every trip, so a program with
one returns only when the loop is done, and the GPU drains between trips. The collection schedule is deterministic
(floor(acc) steps with acc += 1 / utd), so the host computes each iteration's count (StepCounts) and calls one of two
compiled actor programs with a static loop bound. Dispatches then return right after enqueueing, the actor runs from its
own thread, and the host queues each program before its inputs exist (at most one iteration ahead).
"""

from __future__ import annotations

import dataclasses
import math
import os
import queue
import threading
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import buffer as rb
from . import nets, trainer_class
from .algorithm import flat32, widths
from .config import Config
from .runtime import RunLog, compile_all, freeze_gc, make_unflat, run_pipelined, write_json
from .sim import Sim

ACTOR_KEYS = ("env", "obs_hist", "buf", "pi_b", "wm", "wm_b", "opt_wm", "step_acc", "sim_steps", "decay_at", "dec_nd", "dec_rs")
ACTOR_KEYS_AWL = ("env", "obs_hist", "buf", "pi_b", "wm_b", "step_acc", "sim_steps", "decay_at", "dec_nd", "dec_rs")
LEARNER_KEYS = ("pi", "opt_pi", "avg", "avg_n", "critic", "critic_b", "opt_v", "nonfinite")
CRITIC_KEYS = ("critic", "critic_b", "opt_v")
FP_CHUNK = 25  # FP / FP1: iterations per trainer program (probe sweeps start at multiples of eprobe_every = 125)


# ---------------------------------------------------------------- packing trees into flat vectors
def _is_w(path):
    return getattr(path[-1], "key", None) == "w"


def pack(tree):
    """(bf16 weight matrices, fp32 everything else) as two flat vectors. Always a fresh buffer: an output that is also a
    carry leaf could share its buffer with it, and the next program donates the carry."""
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    return (jnp.concatenate([x.astype(jnp.bfloat16).ravel() for p, x in leaves if _is_w(p)]),
            jnp.concatenate([x.astype(jnp.float32).ravel() for p, x in leaves if not _is_w(p)]))  # fmt: skip


def make_unpack(template):
    """Inverse of pack() for trees shaped like ``template`` (weights come back bf16: every matmul casts them anyway)."""
    leaves, tdef = jax.tree_util.tree_flatten_with_path(template)
    spec = [(_is_w(p), x.shape, int(np.prod(x.shape))) for p, x in leaves]

    def unpack(packed):
        wv, ov = packed
        out, iw, io = [], 0, 0
        for is_w, shape, n in spec:
            if is_w:
                out.append(wv[iw : iw + n].reshape(shape))
                iw += n
            else:
                out.append(ov[io : io + n].reshape(shape))
                io += n
        return jax.tree_util.tree_unflatten(tdef, out)

    return unpack


def learner_options(cfg):
    """--learner_xla "key:value,...": compile options for the learner programs only (unlike XLA_FLAGS not applied to the
    programs with MuJoCo Warp physics). CONCURRENT command-buffer scheduling is ~3 % faster on the learners and 5 % slower on
    the physics."""
    opts = dict(x.split(":", 1) for x in cfg.learner_xla.split(",") if x)
    return {"compiler_options": opts} if opts else {}


def sds(tree, dev):
    """Abstract values of ``tree`` placed on ``dev`` (to compile a program for inputs that do not exist yet)."""
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=jax.sharding.SingleDeviceSharding(dev)), tree)


class StepCounts:
    """BlinkMimic's collection schedule on the host (acc += 1 / utd; floor(acc) steps per iteration), in the float32
    arithmetic of the device's step_acc. The actors compile one program per count that occurs (2 and 3 for utd 0.4) and
    take the count as a static loop bound; each actor also reports whether the device's own count agrees (n_mismatch)."""

    def __init__(self, utd, step_acc=0.0):
        self.inc, self.acc = np.float32(1.0 / utd), np.float32(step_acc)
        acc, seen = self.acc, set()
        for _ in range(4096):
            acc = np.float32(acc + self.inc)
            seen.add(int(np.floor(acc)))
            acc = np.float32(acc - np.float32(np.floor(acc)))
        self.counts = tuple(sorted(seen))

    def next(self):
        acc = np.float32(self.acc + self.inc)
        n = int(np.floor(acc))
        self.acc = np.float32(acc - np.float32(n))
        return n


# ---------------------------------------------------------------- probers
class Prober:
    """Probe sweeps of the averaged policy on a dedicated GPU (BlinkMimic's eprobe burst, sweep steps and finish), one
    dispatched program per sweep, from a worker thread: the sweep's dispatch returns only when the sweep is nearly done, and
    from the training thread it would stall training for the whole sweep. The step loop always runs lead_max + tile steps:
    probes past the end of their tile keep stepping with the frame capped and record nothing, as BlinkMimic's do."""

    def __init__(self, xm, dev, key, unpack):
        c = self.cfg = xm.cfg
        self.xm, self.dev = xm, dev
        with jax.default_device(dev):
            self.motion = jax.device_put(xm.motion, dev)
            self.penv = Sim(xm.mj_model, xm.meta, xm.P, xm.mdp, eval_mode=True, gamma=c.gamma, device=dev)
            self.model, pst = self.penv.init(key, self.motion)
        P, nb = xm.P, xm.motion.num_bins
        self.state = jax.device_put({"pst": pst, "hist": jnp.zeros((P, c.hist, xm.obs_dim)), "fail_f": jnp.full((P,), -1, jnp.int32),
                                     "ep_fail": jnp.zeros(nb, jnp.float32)}, dev)  # fmt: skip
        self.unpack = unpack
        self.sweep_j = jax.jit(self._sweep, donate_argnums=(0,))
        self.busy, self.n_sweeps, self.last_fails = False, 0, -1
        self.rng = np.random.default_rng(c.seed + 1234)
        self.todo, self.done = queue.Queue(), queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            req = self.todo.get()
            if req is None:
                return
            snap, lead = req
            try:
                snap = jax.device_put(snap, self.dev, may_alias=False)
                self.state, fq, has, nf = self.sweep_j(self.state, snap, jnp.int32(lead), self.model, self.motion)
                self.done.put((fq, bool(has), int(nf)))  # blocks this thread only
            except Exception as e:  # noqa: BLE001  (re-raised by poll)
                self.done.put(e)

    def _sweep(self, s, snap, lead, model, motion):
        """One sweep: every probe resets on the reference ``lead`` frames before its tile, runs to the tile's end, and its
        first failure counts towards the focus distribution (decayed counts per bin, spread over the eprobe_win bins
        before each failure)."""
        c, xm, penv = self.cfg, self.xm, self.penv
        sp, sb = self.unpack(snap)
        cid = motion.clip_id[xm.ep_tile]
        f0 = jnp.maximum(xm.ep_tile - lead, motion.clip_start[cid])
        stop = jnp.minimum(xm.ep_tile + c.eprobe_tile, motion.clip_end[cid])
        fcap = jnp.minimum(stop, motion.clip_end[cid] - 1)
        pst = {**s["pst"], "done_eval": jnp.zeros(xm.P, bool), "failed_eval": jnp.zeros(xm.P, bool)}
        pst = penv.reset_masked(motion, pst, jnp.ones(xm.P, bool), jax.random.PRNGKey(0), frames=f0)
        o = penv.observe(motion, pst)
        hist0 = jnp.broadcast_to(o[:, None], s["hist"].shape)

        def step(_, st):
            pst, hist, fail_f = st
            o = penv.observe(motion, pst)
            h = jnp.concatenate((hist[:, 1:], o[:, None]), 1)
            a = nets.pi_apply(sp, sb, h, motion.goal(pst["f"], c.goal_frames), xm.ref)
            alive = ~pst["done_eval"]
            f_now = pst["f"]
            pst, out = penv.step_core(model, motion, pst, a)
            fail_f = jnp.where(alive & (out["term"] | ~jnp.isfinite(a).all(-1)), f_now, fail_f)
            pst = {**pst, "done_eval": pst["done_eval"] | (pst["f"] >= stop), "f": jnp.minimum(pst["f"], fcap)}
            return pst, h, fail_f

        pst, hist, ff = jax.lax.fori_loop(0, c.eprobe_lead_max + c.eprobe_tile, step, (pst, hist0, jnp.full((xm.P,), -1, jnp.int32)))
        ep_fail = s["ep_fail"] * c.eprobe_decay
        ep_fail = ep_fail.at[motion.frame_bin[jnp.maximum(ff, 0)]].add((ff >= 0).astype(jnp.float32))
        w, q = ep_fail, ep_fail
        nb = motion.num_bins
        for _ in range(c.eprobe_win):
            w = w[xm.ep_next] * (xm.ep_next != np.arange(nb)).astype(np.float32)
            q = q + w
        out = {"pst": pst, "hist": hist, "fail_f": ff, "ep_fail": ep_fail}
        return out, q / jnp.maximum(q.sum(), 1e-12), ep_fail.sum() > 0, (ff >= 0).sum()

    def compile_job(self, snapshot):
        """A runtime.compile_all job that compiles the sweep program for snapshots shaped like ``snapshot`` (lowered now)."""
        lowered = self.sweep_j.lower(self.state, sds(snapshot, self.dev), jnp.int32(0), self.model, self.motion)

        def job():
            self.sweep_j = lowered.compile()

        return job

    def idle(self):
        return not self.busy

    def start(self, avg_snapshot):
        """Queue a sweep of ``avg_snapshot`` (the averaged policy, packed as ``unpack`` expects; on any device)."""
        lead = int(self.rng.integers(self.cfg.eprobe_lead_min, self.cfg.eprobe_lead_max + 1))
        self.busy = True
        self.todo.put((avg_snapshot, lead))

    def poll(self):
        """(focus distribution, any failures, number of failures) once the queued sweep is done, else None."""
        try:
            res = self.done.get_nowait()
        except queue.Empty:
            return None
        if isinstance(res, Exception):
            raise res
        self.busy, self.n_sweeps, self.last_fails = False, self.n_sweeps + 1, res[2]
        return res

    def close(self):
        """Let a sweep in flight finish and stop the worker: its dispatch runs MuJoCo Warp's Python callbacks, and the
        interpreter shutting down under them would abort the process."""
        if self.busy:
            self.done.get()
            self.busy = False
        self.todo.put(None)


def _prober_child(cfg_dict, todo, done, gpu, packed):
    """The prober process (FP1, AL, AWL): builds the probe simulator and the sweep program on a GPU it shares with a
    trainer process (no memory preallocation, it needs a few GB), then runs one sweep per request. Requests are (averaged
    policy, lead), the policy as one float32 vector (flat32) or, with ``packed``, as pack()'s (bf16 weights, float32 rest);
    replies are (focus distribution, any failures, number of failures)."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:  # only the shared GPU: no CUDA context on the others
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[gpu]
    try:
        cfg = Config(**cfg_dict)
        xm = trainer_class(cfg)(dataclasses.replace(cfg, num_envs=8), probes=False)
        pi, pi_b = nets.pi_init(jax.random.PRNGKey(0), xm.obs_dim, xm.goal_dim, xm.J, cfg.hist, enc_hidden=widths(cfg.enc_hidden),
                                dec_hidden=widths(cfg.dec_hidden))  # fmt: skip
        template = (pi, pi_b)
        prober = Prober(xm, jax.devices()[0], jax.random.PRNGKey(cfg.seed + 99), make_unpack(template) if packed else make_unflat(template))
        prober.compile_job(jax.eval_shape(pack if packed else flat32, template))()
        done.put("ready")
        while True:
            req = todo.get()
            if req is None:
                return
            snap, lead = req
            snap = jax.device_put(tuple(snap) if packed else snap)
            prober.state, fq, has, nf = prober.sweep_j(prober.state, snap, jnp.int32(lead), prober.model, prober.motion)
            done.put((np.asarray(fq), bool(has), int(nf)))
    except Exception as e:  # noqa: BLE001  (re-raised in the trainer by poll / wait_ready)
        import traceback

        done.put(RuntimeError(f"prober process: {e}\n{traceback.format_exc()}"))


class ProberProcess:
    """The prober as a second process on a trainer's GPU. Under CUDA MPS the two processes' kernels run concurrently (the
    probes' small, latency-bound kernels fill SMs the trainer leaves idle); without MPS the GPU time-slices between them.
    Same interface as Prober; snapshots and focus distributions travel through the host."""

    def __init__(self, cfg, gpu, packed):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.cfg = cfg
        self.todo, self.done = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_prober_child, args=(dataclasses.asdict(cfg), self.todo, self.done, gpu, packed), daemon=True)
        self.proc.start()
        self.busy, self.n_sweeps, self.last_fails = False, 0, -1
        self.rng = np.random.default_rng(cfg.seed + 1234)

    def wait_ready(self):
        msg = self.done.get()
        if isinstance(msg, Exception):
            raise msg
        assert msg == "ready", msg

    def compile_job(self, snapshot):
        return self.wait_ready  # the child compiles its sweep program itself; this waits until it is done

    def idle(self):
        return not self.busy

    def start(self, avg_snapshot):
        lead = int(self.rng.integers(self.cfg.eprobe_lead_min, self.cfg.eprobe_lead_max + 1))
        self.busy = True
        self.todo.put((jax.tree.map(np.asarray, jax.device_get(avg_snapshot)), lead))

    def poll(self):
        try:
            res = self.done.get_nowait()
        except queue.Empty:
            return None
        if isinstance(res, Exception):
            raise res
        self.busy, self.n_sweeps, self.last_fails = False, self.n_sweeps + 1, res[2]
        return res

    def close(self):
        """Let a sweep in flight finish, then end the child process."""
        if self.busy:
            self.done.get()
            self.busy = False
        self.todo.put(None)
        self.proc.join(timeout=60)
        if self.proc.is_alive():
            self.proc.terminate()


# ---------------------------------------------------------------- entry
def run(cfg: Config):
    layout = cfg.layout
    assert layout != "1GPU", "the 1GPU layout runs in single_gpu.py"
    probing = cfg.eprobe_share > 0
    prober = None
    if probing and layout in ("FP1", "AL", "AWL"):  # start the prober process first: it builds and compiles while we do
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get("XMIMIC_TRAINER_MEM_FRACTION", "0.8")  # room for it
        gpu = cfg.prober_gpu if cfg.prober_gpu >= 0 else (0 if layout == "FP1" else 1)
        prober = ProberProcess(cfg, gpu=gpu, packed=layout != "FP1")
    devs = jax.devices()
    assert len(devs) >= cfg.gpus, f"layout {layout} needs {cfg.gpus} GPUs, found {len(devs)}"
    os.makedirs(cfg.log_dir, exist_ok=True)
    meta = {**dataclasses.asdict(cfg), "gpus": cfg.gpus}
    write_json(os.path.join(cfg.log_dir, "args.json"), meta)
    t_proc = time.time()
    with jax.default_device(devs[0]):
        xm = trainer_class(cfg)(cfg, probes=False)  # probe sweeps, if any, run on a prober
        consts, carry = xm.init(jax.random.PRNGKey(cfg.seed))
    if probing and layout in ("FP", "ALP", "AWLP"):  # a prober GPU: the last one of the layout
        unpack_snap = make_unflat(carry["avg"]) if layout == "FP" else make_unpack(carry["avg"])
        prober = Prober(xm, devs[cfg.gpus - 1], jax.random.PRNGKey(cfg.seed + 99), unpack_snap)

    def prober_hand_off(it_done, snapshot, set_into, span=1):
        """After iteration ``it_done``, the last of ``span`` iterations since the previous call: apply a finished sweep, and
        start the next one if one fell due in those iterations (every eprobe_every from eprobe_start; a sweep that falls
        due while the prober is still busy is skipped). Returns the log columns."""
        if prober is None:
            return None
        res = prober.poll()
        if res is not None and res[1]:
            set_into(res[0])
        due = any(t >= cfg.eprobe_start and t % cfg.eprobe_every == 0 for t in range(it_done - span + 1, it_done + 1))
        if prober.idle() and due:
            prober.start(snapshot)
        return {"probe_sweeps": prober.n_sweeps, "probe_fails": prober.last_fails}

    def set_focus(state, fq):
        """The prober's focus distribution into the reset sampler (fresh committed arrays: the compiled program is reused)."""
        env = state["env"]
        return {**state, "env": {**env, "focus_q": jax.device_put(fq, env["focus_q"].sharding),
                                 "focus_share": jax.device_put(np.float32(cfg.eprobe_share), env["focus_share"].sharding)}}  # fmt: skip

    ctx = dict(cfg=cfg, xm=xm, consts=consts, carry=carry, devs=devs, prober=prober, hand_off=prober_hand_off,
               set_focus=set_focus, meta=meta, extra_cols=("probe_sweeps", "probe_fails") if prober is not None else (),
               t_proc=t_proc)  # fmt: skip
    del carry
    if layout in ("FP", "FP1"):
        _run_fp(**ctx)
    elif layout in ("AL", "ALP"):
        _run_al(**ctx)
    else:
        _run_awl(**ctx)


def _setup_line(cfg, devs, compile_s, t_proc, t_c, prober, xm):
    print(f"setup {t_c - t_proc:.1f} s | layout {cfg.layout}: {devs[:cfg.gpus]} | aot compile {compile_s:.1f} s (excluded from the "
          f"training clock)" + (f" | prober: {xm.P} probe worlds" if prober is not None else ""), flush=True)
    write_json(os.path.join(cfg.log_dir, "compile.json"), {"compile_s": compile_s})


def _finish(prober, t0, it):
    print(f"done: {it} iterations in {(time.time() - t0) / 60:.2f} min", flush=True)
    if prober is not None:
        prober.close()


def _policy_learner(xm):
    """The policy learner's iteration (AL, ALP, AWL without critic offload): BPTT on the actor's batch of the previous
    iteration with the world model it sent, then the policy EMA and the critic."""

    def learner(consts, cL, msg):
        it = cL["it"]
        itf = it.astype(jnp.float32)
        lr = xm.lr_at(itf)
        pi_lr = lr * xm.pi_mult(itf, msg["decay_at"].astype(jnp.float32))
        key, k_p = jax.random.split(cL["key"])
        tmp = {**cL, "key": key, "wm": msg["wm"], "wm_b": msg["wm_b"], "pi_b": msg["pi_b"]}
        tmp, pstats = xm.policy_update(consts, tmp, k_p, lr, pi_lr, batch=msg["batch"])
        out = {k: tmp[k] for k in cL}
        out["it"] = it + 1
        return out, pack(tmp["pi"]), pack(tmp["avg"]), {**pstats, "lr": lr, "pi_lr": pi_lr, "nonfinite": tmp["nonfinite"].astype(jnp.float32)}

    return learner


def _stack(acc):
    return jax.tree.map(lambda *x: jnp.stack(x), *acc)  # on each leaf's device


# ---------------------------------------------------------------- FP, FP1: the 1GPU trainer and a prober
def _run_fp(cfg, xm, consts, carry, devs, prober, hand_off, set_focus, meta, extra_cols, t_proc):
    t_c = time.time()
    warm = xm.make_warmup(cfg.warmup_steps)
    carry_aval = jax.eval_shape(warm, consts, carry)[0]
    jobs = [warm.lower(consts, carry).compile, xm.make_chunk(FP_CHUNK).lower(consts, carry_aval).compile]
    if prober is not None:
        jobs.append(prober.compile_job(jax.eval_shape(flat32, carry["avg"])))
    warm_c, chunk_c = compile_all(jobs)[:2]
    _setup_line(cfg, devs, time.time() - t_c, t_proc, t_c, prober, xm)

    freeze_gc()
    t0 = time.time()
    carry, _ = warm_c(consts, carry)
    jax.block_until_ready(carry)
    log = RunLog(xm, cfg, t0, carry["avg"], meta, extra_cols)
    state = {"carry": carry, "it": 0}
    del carry

    def dispatch():
        state["carry"], st, avg_flat = chunk_c(consts, state["carry"])
        state["it"] += FP_CHUNK
        return state["it"], st, avg_flat

    def on_done(it_done, avg_flat):  # program it_done is finished: its averaged policy is the snapshot of a sweep starting now
        return hand_off(it_done, avg_flat, lambda fq: state.update(carry=set_focus(state["carry"], fq)), span=FP_CHUNK)

    _finish(prober, t0, run_pipelined(dispatch, log, cfg.iterations, cfg.max_minutes, on_done))


# ---------------------------------------------------------------- AL, ALP: actor | policy learner (| prober)
def _run_al(cfg, xm, consts, carry, devs, prober, hand_off, set_focus, meta, extra_cols, t_proc):
    devA, devL = devs[0], devs[1]
    unpack_pi = make_unpack(carry["pi"])
    learner = _policy_learner(xm)

    def actor(consts, cA, pi_pk, n):
        """Actor iteration: ``n`` collection steps (static, from StepCounts) with the policy ``pi_pk`` of the previous
        learner iteration, the LR-decay trigger, the world-model updates, and the next imagination batch."""
        it = cA["it"]
        lr = xm.lr_at(it.astype(jnp.float32))
        key, k_c, k_w, _, k_b = jax.random.split(cA["key"], 5)
        acc = cA["step_acc"] + 1.0 / cfg.utd
        n_dev = jnp.floor(acc).astype(jnp.int32)
        tmp = {**cA, "key": key, "step_acc": acc - n, "pi": unpack_pi(pi_pk)}
        tmp, cstats = xm.collect(consts, tmp, k_c, n)
        tmp = xm.decay_tick(tmp, it, cstats)  # the LR-decay trigger needs the collection stats: decided here, sent along
        tmp, wstats = xm.wm_updates(tmp, k_w, lr)
        msg = {"batch": xm.sample_policy_batch(consts, tmp["buf"], k_b), "wm": tmp["wm"], "wm_b": tmp["wm_b"], "pi_b": tmp["pi_b"],
               "decay_at": tmp["decay_at"] + 0}  # fmt: skip
        out = {k: tmp[k] for k in cA}
        out["it"] = it + 1
        stats = {"wm": wstats["wm"][-1], "wm8": wstats["wm8"][-1], "n_coll": jnp.asarray(n, jnp.float32), **cstats,
                 "sim_steps": tmp["sim_steps"], "decay_at": tmp["decay_at"].astype(jnp.float32),
                 "n_mismatch": (n_dev != n).astype(jnp.float32)}  # fmt: skip
        return out, msg, stats

    t_c = time.time()
    cA = {k: carry[k] for k in ACTOR_KEYS}
    cA["key"], cA["it"] = carry["key"], carry["it"]
    cL = jax.device_put({**{k: carry[k] for k in LEARNER_KEYS}, "key": jax.random.fold_in(carry["key"], 7), "it": carry["it"]}, devL)
    consts_L = jax.device_put({"motion": consts["motion"]}, devL)
    pi_w = jax.device_put(jax.jit(pack)(cL["pi"]), devA)
    warm_in = {**cA, "pi": unpack_pi(pi_w)}
    sched = StepCounts(cfg.utd, float(jax.device_get(cA["step_acc"])))  # the warm-up does not touch step_acc
    actor_j = jax.jit(actor, static_argnums=(3,), donate_argnums=(1,))
    msg_aval = jax.eval_shape(partial(actor, n=sched.counts[0]), consts, cA, pi_w)[1]
    jobs = [xm.make_warmup(cfg.warmup_steps).lower(consts, warm_in).compile]
    jobs += [actor_j.lower(consts, cA, pi_w, n).compile for n in sched.counts]
    jobs.append(partial(jax.jit(learner, donate_argnums=(1,)).lower(consts_L, cL, sds(msg_aval, devL)).compile, **learner_options(cfg)))
    if prober is not None:
        jobs.append(prober.compile_job(jax.eval_shape(pack, cL["avg"])))
    out = compile_all(jobs)
    warm_c, actor_c, learner_c = out[0], dict(zip(sched.counts, out[1:])), out[1 + len(sched.counts)]
    flat_j = jax.jit(flat32)
    avg_template = carry["avg"]
    del carry
    _setup_line(cfg, devs, time.time() - t_c, t_proc, t_c, prober, xm)

    freeze_gc()
    t0 = time.time()
    warm_out, _ = warm_c(consts, warm_in)
    cA = {k: warm_out[k] for k in cA}
    cA, msg, _ = actor_c[sched.next()](consts, cA, pi_w)  # actor iteration 0; the learner runs one iteration behind
    msg = jax.device_put(msg, devL)
    log = RunLog(xm, cfg, t0, avg_template, meta, extra_cols)

    # The actor runs in its own thread, and only this thread touches the actor's GPU: MuJoCo Warp re-captures its CUDA
    # graphs when buffer addresses change, and an operation on that GPU from another thread during a capture invalidates
    # it. The main thread runs the learner, the transfers to it and the bookkeeping.
    q_pi, q_msg, q_focus = queue.Queue(), queue.Queue(), queue.Queue()
    actor_state = {"cA": cA, "error": None}

    def actor_loop():
        try:
            while True:
                pi_in = q_pi.get()
                if pi_in is None:
                    return
                while not q_focus.empty():
                    actor_state["cA"] = set_focus(actor_state["cA"], q_focus.get())
                actor_state["cA"], msg_new, sA = actor_c[sched.next()](consts, actor_state["cA"], jax.device_put(pi_in, devA))
                q_msg.put((jax.device_put(msg_new, devL), jax.device_put(sA, devL)))
        except Exception as e:  # noqa: BLE001  (re-raised by the main loop)
            actor_state["error"] = e
            q_msg.put(None)

    th = threading.Thread(target=actor_loop, daemon=True)
    th.start()
    q_pi.put(pi_w)  # actor iteration 1 acts with the initial policy; the learner runs one iteration behind
    it, acc, pending, stop = 0, [], None, False
    prev_sL, n_mismatch = None, 0.0
    while it < cfg.iterations and not stop:
        cL, pi_new, avg_new, sL = learner_c(consts_L, cL, msg)  # on actor iteration it-1's message
        if it + 1 < cfg.iterations:
            q_pi.put(pi_new)  # actor iteration it+1 acts with this update's policy (the actor thread copies it over)
        res = q_msg.get()  # actor iteration it
        if res is None:
            raise actor_state["error"]
        msg, sA = res
        acc.append({**sA, **sL})
        it += 1
        if pending is not None:  # the previous log interval, now that this iteration is queued behind it
            if log.add(*pending[:3], extra=pending[3]) >= cfg.max_minutes > 0:
                print(f"time budget {cfg.max_minutes} min reached at it {pending[0]}", flush=True)
                stop = True
            pending = None
        extra = hand_off(it, avg_new, q_focus.put)
        if it % cfg.log_every == 0 or it >= cfg.iterations or stop:
            pending = (it, _stack(acc), flat_j(cL["avg"]), extra)  # queued before the next learner call donates cL
            log.submitted(it, pending[1])
            n_mismatch = n_mismatch + jnp.sum(pending[1]["n_mismatch"])
            acc = []
        if prev_sL is not None:  # run-ahead limit: the learner has finished the previous iteration
            jax.block_until_ready(prev_sL)
        prev_sL = sL["pi"]
    q_pi.put(None)
    th.join()
    if pending is not None:
        log.add(*pending[:3], extra=pending[3])
    log.close()
    _finish(prober, t0, it)
    print(f"collection counts: host schedule != device step_acc in {int(n_mismatch)} iterations", flush=True)


# ---------------------------------------------------------------- AWL, AWLP: actor | world-model learner | policy learner (| prober)
def _run_awl(cfg, xm, consts, carry, devs, prober, hand_off, set_focus, meta, extra_cols, t_proc):
    """Iteration i: the actor collects with policy i-1 and samples the imagination batch; the world-model learner appends
    the actor's transitions of iteration i-1 to its replica of the replay buffer (the actor sends the rows it wrote) and
    runs the iteration's world-model updates; the policy learner updates on the actor's batch of iteration i-1 with the
    world model of iteration i-1. Relative to AL the world model is one iteration older."""
    assert prober is not None or cfg.eprobe_share <= 0 or cfg.layout == "AWLP"
    devA, devW, devL = devs[0], devs[1], devs[2]
    n_max = math.ceil(1.0 / cfg.utd - 1e-9)  # collection steps per iteration: floor(acc) <= ceil(1 / utd)
    f32 = jnp.float32
    unpack_pi, unpack_wm = make_unpack(carry["pi"]), make_unpack(carry["wm"])
    learner = _policy_learner(xm)
    co = bool(cfg.critic_offload)
    if co:
        assert hasattr(xm, "critic_fit"), "--critic_offload needs the fast learner (or --critic_offload 0)"

    def actor(consts, cA, pi_pk, n):
        """Actor iteration: ``n`` collection steps (static, from StepCounts) with policy i-1, the LR-decay trigger, the
        rows it wrote for the world-model learner's replica, and the next imagination batch for the policy learner."""
        it = cA["it"]
        key, k_c, k_b = jax.random.split(cA["key"], 3)
        acc = cA["step_acc"] + 1.0 / cfg.utd
        n_dev = jnp.floor(acc).astype(jnp.int32)
        ptr0 = cA["buf"]["ptr"]
        tmp = {**cA, "key": key, "step_acc": acc - n, "pi": unpack_pi(pi_pk)}
        tmp, cstats = xm.collect(consts, tmp, k_c, n)
        tmp = xm.decay_tick(tmp, it, cstats)
        # the n_max rows from the write head on: the n new ones, then unchanged rows that the replica holds too
        idx = (ptr0 + jnp.arange(n_max)) % tmp["buf"]["s"].shape[0]
        msg_w = {"rows": {k: tmp["buf"][k][idx] for k in rb.FIELDS}, "n": jnp.asarray(n, jnp.int32),
                 "wm_b": jax.tree.map(lambda x: x + 0, tmp["wm_b"])}  # fmt: skip
        msg_l = {"batch": xm.sample_policy_batch(consts, tmp["buf"], k_b), "pi_b": jax.tree.map(lambda x: x + 0, tmp["pi_b"]),
                 "decay_at": tmp["decay_at"] + 0}  # fmt: skip
        out = {k: tmp[k] for k in cA}
        out["it"] = it + 1
        stats = {"n_coll": jnp.asarray(n, f32), **cstats, "sim_steps": tmp["sim_steps"], "decay_at": tmp["decay_at"].astype(f32),
                 "n_mismatch": (n_dev != n).astype(f32)}  # fmt: skip
        return out, msg_w, msg_l, stats

    def wm_learner(cW, msg_w):
        it = cW["it"]
        lr = xm.lr_at(it.astype(f32))
        key, k_w = jax.random.split(cW["key"])
        buf = cW["buf"]
        idx = (buf["ptr"] + jnp.arange(n_max)) % buf["s"].shape[0]
        buf = {**{k: buf[k].at[idx].set(msg_w["rows"][k]) for k in rb.FIELDS}, "ptr": (buf["ptr"] + msg_w["n"]) % buf["s"].shape[0],
               "size": jnp.minimum(buf["size"] + msg_w["n"], buf["s"].shape[0])}  # fmt: skip
        tmp, wstats = xm.wm_updates({"buf": buf, "wm": cW["wm"], "wm_b": msg_w["wm_b"], "opt_wm": cW["opt_wm"]}, k_w, lr)
        out = {"buf": tmp["buf"], "wm": tmp["wm"], "opt_wm": tmp["opt_wm"], "key": key, "it": it + 1}
        return out, {"wm": pack(tmp["wm"]), "wm_b": jax.tree.map(lambda x: x + 0, msg_w["wm_b"])}, {"wm": wstats["wm"][-1], "wm8": wstats["wm8"][-1]}

    def pol_learner(consts, cL, msg_l, msg_wm):
        return learner(consts, cL, {**msg_l, "wm": unpack_wm(msg_wm["wm"]), "wm_b": msg_wm["wm_b"]})

    # critic offload: the critic phase (normalizer, GAE targets, 20 regression steps) runs on the world-model learner's
    # GPU; policy update i bootstraps with the critic fitted on policy update i-2's imagination (i-1 without the offload)
    def pol_learner_co(consts, cL, msg_l, msg_wm, msg_c):
        it = cL["it"]
        itf = it.astype(f32)
        lr = xm.lr_at(itf)
        pi_lr = lr * xm.pi_mult(itf, msg_l["decay_at"].astype(f32))
        key, k_p = jax.random.split(cL["key"])
        tmp = {**cL, "key": key, "wm": unpack_wm(msg_wm["wm"]), "wm_b": msg_wm["wm_b"], "pi_b": msg_l["pi_b"], **msg_c}
        tmp, pstats, crit = xm.policy_update(consts, tmp, k_p, lr, pi_lr, batch=msg_l["batch"], critic_out=True)
        out = {k: tmp[k] for k in cL}
        out["it"] = it + 1
        return out, pack(tmp["pi"]), pack(tmp["avg"]), {**pstats, "lr": lr, "pi_lr": pi_lr, "nonfinite": tmp["nonfinite"].astype(f32)}, crit

    def crit_learner(consts_W, cC, crit):
        X = jnp.concatenate((crit["feat"], consts_W["motion"].goal(crit["F"], cfg.goal_frames)), -1)
        critic, critic_b, opt_v, v_losses = xm.critic_fit(cC, X, crit["r"], crit["d"], crit["k"], crit["lr"])
        fresh = lambda t: jax.tree.map(lambda x: x + 0, t)  # noqa: E731  (the message must not alias the donated carry)
        return {"critic": critic, "critic_b": critic_b, "opt_v": opt_v}, {"critic": fresh(critic), "critic_b": fresh(critic_b)}, {"v": v_losses[-1]}

    t_c = time.time()
    put = jax.device_put
    cA = {k: carry[k] for k in ACTOR_KEYS_AWL}
    cA["key"], cA["it"] = carry["key"], carry["it"]
    cL = put({**{k: carry[k] for k in LEARNER_KEYS if not (co and k in CRITIC_KEYS)}, "key": jax.random.fold_in(carry["key"], 7),
              "it": carry["it"]}, devL)  # fmt: skip
    if co:
        cC = put({k: carry[k] for k in CRITIC_KEYS}, devW)
        consts_W = put({"motion": consts["motion"]}, devW)
        msg_c = put({"critic": carry["critic"], "critic_b": carry["critic_b"]}, devL)
    cW = put({"buf": carry["buf"], "wm": carry["wm"], "opt_wm": carry["opt_wm"], "key": jax.random.fold_in(carry["key"], 11), "it": carry["it"]}, devW)
    consts_L = put({"motion": consts["motion"]}, devL)
    pi_w = put(jax.jit(pack)(cL["pi"]), devA)
    warm_in = {**cA, "pi": unpack_pi(pi_w)}
    sched = StepCounts(cfg.utd, float(jax.device_get(cA["step_acc"])))  # the warm-up does not touch step_acc
    actor_j = jax.jit(actor, static_argnums=(3,), donate_argnums=(1,))
    _, mw_aval, ml_aval, _ = jax.eval_shape(partial(actor, n=sched.counts[0]), consts, cA, pi_w)
    mwm_aval = jax.eval_shape(wm_learner, cW, sds(mw_aval, devW))[1]
    lopts = learner_options(cfg)
    jobs = [xm.make_warmup(cfg.warmup_steps).lower(consts, warm_in).compile]
    jobs += [actor_j.lower(consts, cA, pi_w, n).compile for n in sched.counts]
    jobs.append(partial(jax.jit(wm_learner, donate_argnums=(0,)).lower(cW, sds(mw_aval, devW)).compile, **lopts))
    if co:
        jobs.append(partial(jax.jit(pol_learner_co, donate_argnums=(1,)).lower(consts_L, cL, sds(ml_aval, devL), sds(mwm_aval, devL),
                                                                             msg_c).compile, **lopts))  # fmt: skip
        crit_aval = jax.eval_shape(pol_learner_co, consts_L, cL, sds(ml_aval, devL), sds(mwm_aval, devL), msg_c)[4]
        jobs.append(partial(jax.jit(crit_learner, donate_argnums=(1,)).lower(consts_W, cC, sds(crit_aval, devW)).compile, **lopts))
    else:
        jobs.append(partial(jax.jit(pol_learner, donate_argnums=(1,)).lower(consts_L, cL, sds(ml_aval, devL), sds(mwm_aval, devL)).compile,
                            **lopts))  # fmt: skip
    if prober is not None:
        jobs.append(prober.compile_job(jax.eval_shape(pack, cL["avg"])))
    out = compile_all(jobs)
    k = len(sched.counts)
    warm_c, actor_c, wm_c, pol_c = out[0], dict(zip(sched.counts, out[1 : 1 + k])), out[1 + k], out[2 + k]
    crit_c = out[3 + k] if co else None
    flat_j = jax.jit(flat32)
    rows_j = jax.jit(lambda buf, wm_b: {"rows": {k: buf[k][(buf["ptr"] + jnp.arange(n_max)) % buf["s"].shape[0]] for k in rb.FIELDS},
                                        "n": jnp.zeros((), jnp.int32), "wm_b": jax.tree.map(lambda x: x + 0, wm_b)})  # fmt: skip
    avg_template = carry["avg"]
    del carry
    _setup_line(cfg, devs, time.time() - t_c, t_proc, t_c, prober, xm)

    freeze_gc()
    t0 = time.time()
    warm_out, _ = warm_c(consts, warm_in)
    cA = {k: warm_out[k] for k in cA}
    # the world-model learner's replica starts as the actor's buffer after the warm-up; its first updates append nothing
    cW = {**cW, "buf": put(cA["buf"], devW)}
    cW, msg_wm, _ = wm_c(cW, put(rows_j(cA["buf"], cA["wm_b"]), devW))
    msg_wm = put(msg_wm, devL)
    cA, msg_w, msg_l, _ = actor_c[sched.next()](consts, cA, pi_w)  # actor iteration 0 with the initial policy
    msg_w, msg_l = put(msg_w, devW), put(msg_l, devL)
    log = RunLog(xm, cfg, t0, avg_template, meta, extra_cols)

    # the actor in its own thread, the only one that touches the actor's GPU (see _run_al)
    q_pi, q_msg, q_focus = queue.Queue(), queue.Queue(), queue.Queue()
    actor_state = {"cA": cA, "error": None}

    def actor_loop():
        try:
            while True:
                pi_in = q_pi.get()
                if pi_in is None:
                    return
                while not q_focus.empty():
                    actor_state["cA"] = set_focus(actor_state["cA"], q_focus.get())
                actor_state["cA"], mw, ml, sA = actor_c[sched.next()](consts, actor_state["cA"], put(pi_in, devA))
                q_msg.put((put(mw, devW), put(ml, devL), put(sA, devL)))
        except Exception as e:  # noqa: BLE001  (re-raised by the main loop)
            actor_state["error"] = e
            q_msg.put(None)

    th = threading.Thread(target=actor_loop, daemon=True)
    th.start()
    q_pi.put(pi_w)  # actor iteration 1 acts with the initial policy, as the learner's first update has not happened yet
    it, acc, pending, stop = 0, [], None, False
    host = {k: [] for k in ("pol", "wm", "wait_actor", "rest", "wait_pol")}  # main-thread seconds per iteration
    pc = time.perf_counter
    prev_sL, n_mismatch = None, 0.0
    crit_msg, sC = None, {"v": put(np.zeros((), np.float32), devL)}  # (never the default device: that is the actor's GPU)
    while it < cfg.iterations and not stop:
        t1 = pc()
        if co:  # policy update on actor iteration it-1's batch, bootstrapping with the critic fitted on update it-2
            cL, pi_new, avg_new, sL, crit_new = pol_c(consts_L, cL, msg_l, msg_wm, msg_c)
        else:
            cL, pi_new, avg_new, sL = pol_c(consts_L, cL, msg_l, msg_wm)
        t2 = pc()
        cW, msg_wm_new, sW = wm_c(cW, msg_w)  # world-model updates on the replica with actor iteration it-1's rows
        if co and crit_msg is not None:  # the critic phase of policy update it-1, on the world-model learner's GPU
            cC, msg_c_new, sC_new = crit_c(consts_W, cC, crit_msg)
            msg_c, sC = put(msg_c_new, devL), put(sC_new, devL)
        if co:
            crit_msg = put(crit_new, devW)
        t3 = pc()
        if it + 1 < cfg.iterations and not stop:
            q_pi.put(pi_new)  # actor iteration it+1 acts with this update's policy (copied to its GPU by the actor thread)
        res = q_msg.get()  # actor iteration it
        if res is None:
            raise actor_state["error"]
        msg_w, msg_l, sA = res
        t4 = pc()
        msg_wm = put(msg_wm_new, devL)
        acc.append({**sA, **sW, **sL, **(sC if co else {})})
        it += 1
        if pending is not None:
            if log.add(*pending[:3], extra=pending[3]) >= cfg.max_minutes > 0:
                print(f"time budget {cfg.max_minutes} min reached at it {pending[0]}", flush=True)
                stop = True
            pending = None
        extra = hand_off(it, avg_new, q_focus.put)
        if it % cfg.log_every == 0 or it >= cfg.iterations or stop:
            pending = (it, _stack(acc), flat_j(cL["avg"]), extra)
            log.submitted(it, pending[1])
            n_mismatch = n_mismatch + jnp.sum(pending[1]["n_mismatch"])
            acc = []
        t5 = pc()
        if prev_sL is not None:  # run-ahead limit: the policy learner has finished the previous iteration
            jax.block_until_ready(prev_sL)
        prev_sL = sL["pi"]
        for k, v in zip(host, (t2 - t1, t3 - t2, t4 - t3, t5 - t4, pc() - t5)):
            host[k].append(v)
    q_pi.put(None)
    th.join()
    cA = actor_state["cA"]
    if pending is not None:
        log.add(*pending[:3], extra=pending[3])
    log.close()
    _finish(prober, t0, it)
    print(f"collection counts: host schedule != device step_acc in {int(n_mismatch)} iterations", flush=True)
    print("main-thread ms per iteration (median / mean): " + ", ".join(
        f"{k} {1e3 * float(np.median(v)):.2f} / {1e3 * float(np.mean(v)):.2f}" for k, v in host.items()), flush=True)
    if os.environ.get("XMIMIC_CHECK_REPLICA") == "1":  # the replica must equal the actor's buffer except the last rows
        bA, bW, n_last = jax.device_get(cA["buf"]), jax.device_get(cW["buf"]), int(jax.device_get(msg_w["n"]))
        T = bA["s"].shape[0]
        fresh = (int(bW["ptr"]) + np.arange(n_last)) % T
        keep = np.setdiff1d(np.arange(T), fresh)
        bad = [k for k in rb.FIELDS if not np.array_equal(bA[k][keep], bW[k][keep])]
        ok = not bad and (int(bW["ptr"]) + n_last) % T == int(bA["ptr"])
        print(f"replica check: actor ptr {int(bA['ptr'])} size {int(bA['size'])}, replica ptr {int(bW['ptr'])} size "
              f"{int(bW['size'])}, last n {n_last}: {'replica == actor buffer' if ok else 'MISMATCH ' + str(bad)}", flush=True)
