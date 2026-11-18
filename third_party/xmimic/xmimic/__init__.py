"""XMimic: BlinkMimic's world-model motion tracker for the Unitree G1, rebuilt in JAX with MJX and MuJoCo Warp.

Modules, roughly in the order data flows through them:
  model, motion, kin, rot     the G1 model, the reference motions and their kinematics
  sim, mdp, buffer            the batched simulator, the tracking task and the replay buffer
  nets, optim                 world model, policy and critic; MuonAdam, Adam and the policy weight average
  algorithm                   one training iteration (the reference implementation)
  fast, fast_nets             the same iteration with fewer, larger GPU kernels (the default learner)
  single_gpu, multi_gpu       the layouts: how iterations are dispatched to one or several GPUs
  runtime                     train log, checkpoints, host loop
  evaluate                    checkpoint evaluation with BlinkMimic's protocol
"""


def trainer_class(cfg):
    """The training class for cfg.learner: the fast learner (default) or the reference implementation."""
    if cfg.learner == "fast":
        from .fast import FastXMimic

        return FastXMimic
    from .algorithm import XMimic

    return XMimic
