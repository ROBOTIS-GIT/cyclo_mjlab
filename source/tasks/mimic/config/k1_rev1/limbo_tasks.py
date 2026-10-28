"""Register the current OBS124 limbo training task only."""
from mjlab.tasks.registry import register_mjlab_task
from .limbo_robust_env_cfg import k1_rev1_limbo_robust_env_cfg, k1_rev1_limbo_robust_runner_cfg
from source.tasks.mimic.limbo_robust_runner import RobustRunner

register_mjlab_task(
    task_id="Cyclo-Mimic-K1-Rev1-Limbo-1p2m-RobustObs124",
    env_cfg=k1_rev1_limbo_robust_env_cfg(),
    play_env_cfg=k1_rev1_limbo_robust_env_cfg(play=True),
    rl_cfg=k1_rev1_limbo_robust_runner_cfg(),
    runner_cls=RobustRunner,
)
