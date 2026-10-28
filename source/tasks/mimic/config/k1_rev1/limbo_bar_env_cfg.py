"""One fixed bar, 1.25 m underside, 1.44 m contact-planned travel."""
from dataclasses import fields
import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from source import SRC_PATH
from source.tasks.mimic.mdp import limbo_bar
from .limbo_env_cfg import k1_rev1_limbo_env_cfg, k1_rev1_limbo_runner_cfg


def make_bar():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="frame", mocap=True)
  h = limbo_bar.BAR_UNDERSIDE + limbo_bar.BAR_RADIUS
  body.add_geom(name="crossbar", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=[-0.8, 0, h, 0.8, 0, h], size=[limbo_bar.BAR_RADIUS, 0, 0],
    contype=1, conaffinity=1, rgba=[1, 0.7, 0.1, 1], friction=[0.6, 0.005, 0.0001])
  for side, x in (("left", -0.8), ("right", 0.8)):
    body.add_geom(name=side + "_post", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=[x, 0, 0.025, x, 0, h], size=[0.025, 0, 0],
      contype=1, conaffinity=1, rgba=[0.25, 0.3, 0.35, 1])
  return spec


def k1_rev1_limbo_bar_env_cfg(play=False):
  cfg = k1_rev1_limbo_env_cfg(play=play)
  cfg.episode_length_s = 70.0
  old = cfg.commands["reference_trajectory"]
  ref = limbo_bar.LimboBarCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
  ref.trajectory_file = str(SRC_PATH / "assets/motions/K1_rev1/limbo_bar_v3/limbo.npz")
  cfg.commands["reference_trajectory"] = ref
  cfg.scene.entities["bar"] = EntityCfg(spec_fn=make_bar,
    init_state=EntityCfg.InitialStateCfg(pos=(0, limbo_bar.BAR_Y, 0)))
  cfg.events["bar_pose"] = EventTermCfg(func=limbo_bar.reset_bar, mode="reset")
  cfg.scene.sensors += (ContactSensorCfg(name="limbo_bar_contact",
    primary=ContactMatch(mode="body", pattern="frame", entity="bar"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"), reduce="netforce", history_length=4),)
  for name, func in (("bar_collision", limbo_bar.bar_contact),
                     ("nonfoot_support", limbo_bar.nonfoot_support),
                     ("leave_lane", limbo_bar.leave_lane),
                     ("unsuccessful_end", limbo_bar.unsuccessful_end)):
    cfg.terminations[name] = TerminationTermCfg(func=func)
  cfg.rewards["limbo_full_success"] = RewardTermCfg(func=limbo_bar.completion_reward, weight=10.0)
  return cfg


def k1_rev1_limbo_bar_runner_cfg():
  cfg = k1_rev1_limbo_runner_cfg()
  cfg.experiment_name = "k1_limbo_bar_v3"
  cfg.run_name = "bar125_travel144"
  cfg.algorithm.learning_rate = 1e-4
  cfg.save_interval = 500
  return cfg
