"""All options of a run. The defaults are the configuration behind the published H100 results; every option is also a
command-line flag of train.py (``--name value``).

The algorithm block is BlinkMimic's PRESET[main] (pinned commit d264658) and should stay as it is for comparable results.
"""

from __future__ import annotations

import argparse
import dataclasses

LAYOUTS = ("1GPU", "FP1", "FP", "AL", "ALP", "AWL", "AWLP")
GPUS_PER_LAYOUT = {"1GPU": 1, "FP1": 1, "FP": 2, "AL": 2, "ALP": 3, "AWL": 3, "AWLP": 4}


@dataclasses.dataclass
class Config:
    # ---- algorithm (BlinkMimic PRESET[main])
    num_envs: int = 4096  # simulated worlds
    iterations: int = 3500
    buffer_steps: int = 1000  # replay buffer length in control steps (x num_envs transitions)
    utd: float = 0.4  # update-to-data ratio: 1 / utd collection steps per iteration on average (2, 3, 2, 3, ...)
    horizon: int = 32  # imagined steps per policy update (backpropagation through time)
    goal_frames: int = 10  # future reference frames in the policy's goal
    hist: int = 10  # observation history of the policy
    wm_window: int = 8  # world-model loss: open-loop prediction over windows of this many steps
    wm_batches: int = 5  # world-model updates per iteration
    wm_batch: int = 2048
    pi_batch: int = 2048  # imagined rollouts per policy update
    lr: float = 4e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    wm_hidden: str = "1024,1024,1024"
    enc_hidden: str = "2048,1024,512,256"
    dec_hidden: str = "2048,1024,512,256"
    fail_reset: float = 0.3  # share of resets drawn near recent failures
    pi_beta1: float = 0.9
    pi_clip_value: float = 0.1  # elementwise policy-gradient clip
    pi_ema: float = 0.995  # weight average of the policy; the average is the deployed and evaluated policy
    warmup_steps: int = 64  # collection steps before the first update
    contact_reward: float = 0.25
    wm_mirror: float = 0.5  # share of each world-model batch mirrored left-right
    pi_lr_mult: float = 4.0  # policy LR multiplier until the decay trigger, then ramped down to 1 over lr_decay_len
    lr_decay_ret: float = 0.75  # decay trigger: mean episode return of a log interval reaches this ...
    lr_decay_len: int = 300
    lr_decay_by: int = 1400  # ... or this iteration is reached
    lr_decay_iters: int = 250_000  # cosine LR schedule length
    # ---- evaluation-guided resets: probe sweeps of the averaged policy over all clips under evaluation conditions; the
    #      failures they find become a focus distribution for the training resets (BlinkMimic --eprobe_*)
    eprobe_share: float = 0.15  # share of resets drawn from the focus distribution (0 disables the probes)
    eprobe_tile: int = 300  # one probe world per tile of this many frames (1,489 worlds for the 40 clips)
    eprobe_lead_min: int = 200  # probes start a random lead before their tile
    eprobe_lead_max: int = 450
    eprobe_every: int = 125  # a sweep starts every this many iterations ...
    eprobe_start: int = 1000  # ... from this iteration on
    eprobe_win: int = 10  # failures spread over this many bins before them
    eprobe_decay: float = 0.7  # decay of the failure counts per sweep
    eprobe_async: int = 20  # 1GPU layout: sweep steps per training iteration while a sweep runs
    # ---- run
    log_every: int = 50  # iterations per log line (and per compiled program on one GPU)
    ckpt_every: int = 100
    seed: int = 0
    model_dir: str = ""  # optional compiled scene export and robot metadata
    motion_dir: str = "data/lafan1_g1"
    log_dir: str = "runs/xmimic"
    max_minutes: float = 0.0  # stop after this much training time (0: run all iterations)
    # ---- layout and learner
    layout: str = "1GPU"  # 1GPU | FP1 | FP | AL | ALP | AWL | AWLP (README, "How it gets fast")
    learner: str = "fast"  # fast (xmimic/fast.py) | reference (xmimic/algorithm.py): the same algorithm
    prober_gpu: int = -1  # FP1 / AL / AWL: GPU of the prober process (-1: GPU 0 for FP1, GPU 1 for AL and AWL)
    critic_offload: int = 1  # AWL / AWLP: run the critic fit on the world-model learner's GPU (one iteration older critic)
    learner_xla: str = "xla_gpu_command_buffer_scheduling_mode:CONCURRENT"  # XLA options of the learner programs only
    #                                                                          (AL, ALP, AWL, AWLP), "key:value,..."
    # ---- simulation and memory
    njmax: int = 256  # MuJoCo Warp constraint rows per world (BlinkMimic: 500; 256 overflows a few times per run)
    imag_remat: str = "none"  # none | dots | full: recompute imagination activations in the backward pass to save memory

    @property
    def gpus(self) -> int:
        return GPUS_PER_LAYOUT[self.layout]


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Train XMimic (all options: xmimic/config.py).")
    for f in dataclasses.fields(Config):
        p.add_argument("--" + f.name, type=type(f.default), default=f.default)
    cfg = Config(**vars(p.parse_args(argv)))
    assert cfg.layout in LAYOUTS, f"--layout must be one of {LAYOUTS}"
    assert cfg.learner in ("fast", "reference"), "--learner must be fast or reference"
    assert cfg.imag_remat in ("none", "dots", "full"), "--imag_remat must be none, dots or full"
    return cfg
