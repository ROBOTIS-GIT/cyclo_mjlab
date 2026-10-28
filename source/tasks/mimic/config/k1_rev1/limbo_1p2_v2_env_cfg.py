"""Track the 376-frame contact V2 motion through a physical 1.20m bar."""
from dataclasses import fields
import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from source import SRC_PATH
from source.tasks.mimic.mdp import limbo_1p2_v2
from .limbo_bar_env_cfg import k1_rev1_limbo_bar_env_cfg, k1_rev1_limbo_bar_runner_cfg


def make_bar_1p2_v2():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="frame", mocap=True)
  h = limbo_1p2_v2.BAR_UNDERSIDE + limbo_1p2_v2.BAR_RADIUS
  body.add_geom(name="crossbar", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=[-0.8, 0, h, 0.8, 0, h], size=[limbo_1p2_v2.BAR_RADIUS, 0, 0],
    contype=1, conaffinity=1, rgba=[1, 0.7, 0.1, 1], friction=[0.6, 0.005, 0.0001])
  for side, x in (("left", -0.8), ("right", 0.8)):
    body.add_geom(name=side + "_post", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=[x, 0, 0.025, x, 0, h], size=[0.025, 0, 0],
      contype=1, conaffinity=1, rgba=[0.25, 0.3, 0.35, 1])
  return spec


def k1_rev1_limbo_1p2_v2_env_cfg(play=False):
  cfg = k1_rev1_limbo_bar_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 2048
  cfg.episode_length_s = 9.0
  old = cfg.commands["reference_trajectory"]
  ref = limbo_1p2_v2.Limbo1p2V2CommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
  ref.trajectory_file = str(SRC_PATH / "assets/motions/K1_rev1/limbo_1p2_contact_v2/limbo.npz")
  cfg.commands["reference_trajectory"] = ref
  cfg.scene.entities["bar"] = EntityCfg(spec_fn=make_bar_1p2_v2,
    init_state=EntityCfg.InitialStateCfg(pos=(0, limbo_1p2_v2.BAR_Y, 0)))
  cfg.terminations["unsuccessful_end"] = TerminationTermCfg(func=limbo_1p2_v2.unsuccessful_end)
  cfg.rewards["limbo_full_success"] = RewardTermCfg(func=limbo_1p2_v2.completion_reward, weight=10.0)
  return cfg


def k1_rev1_limbo_1p2_v2_runner_cfg():
  cfg = k1_rev1_limbo_bar_runner_cfg()
  cfg.experiment_name = "k1_limbo_1p2_contact_v2"
  cfg.run_name = "bar120_contact_v2"
  cfg.max_iterations = 30000
  return cfg
