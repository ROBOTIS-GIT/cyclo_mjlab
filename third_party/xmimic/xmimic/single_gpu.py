"""The 1GPU layout: everything on one GPU, ``log_every`` (50) iterations per compiled XLA program.

Every buffer is donated and nothing returns to the host inside a program; the LR-decay trigger is evaluated on the device.
The host dispatches program k+1 before it logs program k, and checkpoints are written from a background thread, so the
GPU never waits for Python. Compilation happens before the training clock starts.
"""

from __future__ import annotations

import dataclasses
import os
import time

import jax

from . import trainer_class
from .config import Config
from .runtime import RunLog, compile_all, freeze_gc, run_pipelined, write_json


def run(cfg: Config):
    os.makedirs(cfg.log_dir, exist_ok=True)
    meta = {**dataclasses.asdict(cfg), "gpus": cfg.gpus}
    write_json(os.path.join(cfg.log_dir, "args.json"), meta)
    t_proc = time.time()
    xm = trainer_class(cfg)(cfg)
    consts, carry = xm.init(jax.random.PRNGKey(cfg.seed))
    jax.block_until_ready(carry)
    print(f"setup {time.time() - t_proc:.1f} s | motion {xm.motion.num_frames} frames, {xm.motion.num_bins} bins | "
          f"goal {xm.goal_dim} critic_in {xm.critic_in} wm_in {xm.wm_in} | devices {jax.devices()}", flush=True)
    # compile both programs, side by side, before the training clock starts (from the persistent cache: a load)
    t_c = time.time()
    warm, chunk = xm.make_warmup(cfg.warmup_steps), xm.make_chunk(cfg.log_every)
    carry_aval = jax.eval_shape(warm, consts, carry)[0]
    lowered = (warm.lower(consts, carry), chunk.lower(consts, carry_aval))
    warm, chunk = compile_all([lambda lw=lw: lw.compile() for lw in lowered])
    compile_s = time.time() - t_c
    print(f"aot compile: {compile_s:.1f} s (excluded from the training clock)", flush=True)
    write_json(os.path.join(cfg.log_dir, "compile.json"), {"compile_s": compile_s})

    freeze_gc()
    t0 = time.time()  # the training clock
    carry, _ = warm(consts, carry)
    jax.block_until_ready(carry)
    print(f"warm-up: {cfg.warmup_steps} collection steps {time.time() - t0:.1f} s", flush=True)
    log = RunLog(xm, cfg, t0, carry["avg"], meta)
    state = {"carry": carry, "it": 0}
    del carry

    def dispatch():
        state["carry"], st, avg_flat = chunk(consts, state["carry"])
        state["it"] += cfg.log_every
        return state["it"], st, avg_flat

    it = run_pipelined(dispatch, log, cfg.iterations, cfg.max_minutes)
    print(f"done: {it} iterations in {(time.time() - t0) / 60:.2f} min", flush=True)
