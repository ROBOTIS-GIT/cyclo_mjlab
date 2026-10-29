"""Register limbo with the standard K1 Rev.1 BeyondMimic environment and PPO."""

from mjlab.tasks.registry import register_mjlab_task

from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from .limbo_env_cfg import k1_rev1_limbo_env_cfg

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Limbo-1p2m",
  env_cfg=k1_rev1_limbo_env_cfg(),
  play_env_cfg=k1_rev1_limbo_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)
