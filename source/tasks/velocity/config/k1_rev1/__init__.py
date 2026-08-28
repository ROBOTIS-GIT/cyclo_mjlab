from mjlab.tasks.registry import register_mjlab_task
from source.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import cyclo_k1_rev1_flat_env_cfg
from .rl_cfg import robotis_k1_ppo_runner_cfg


register_mjlab_task(
  task_id="Cyclo-Velocity-Flat-K1-Rev1-v0",
  env_cfg=cyclo_k1_rev1_flat_env_cfg(),
  play_env_cfg=cyclo_k1_rev1_flat_env_cfg(play=True),
  rl_cfg=robotis_k1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
