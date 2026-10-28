"""Stage 1: track the 30-second K1 limbo reference on flat ground, without a bar."""

from dataclasses import fields

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from source import SRC_PATH
from source.tasks.mimic.mdp import limbo
from .base_env_cfg import k1_rev1_mimic_env_cfg
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from source.assets.robots.limbo_k1_rev1 import get_k1_rev1_cfg as get_limbo_k1_rev1_cfg


def k1_rev1_limbo_env_cfg(play=False):
  cfg = k1_rev1_mimic_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_limbo_k1_rev1_cfg()
  cfg.scene.num_envs = 1 if play else 512
  cfg.episode_length_s = 31.0
  original = cfg.commands["reference_trajectory"]
  ref = limbo.LimboCommandCfg(**{f.name: getattr(original, f.name) for f in fields(original)})
  ref.trajectory_file = str(SRC_PATH / "assets/motions/K1_rev1/limbo_contact_v2/limbo.npz")
  ref.reset_pose_noise = {}
  ref.reset_velocity_noise = {}
  ref.reset_joint_position_noise = (0.0, 0.0)
  ref.start_from_zero = play
  cfg.commands["reference_trajectory"] = ref

  # This reference approaches model joint limits with about a 0.002 rad margin.
  # The dance task's 90% soft limits would clip valid limbo reset poses.
  cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor = 1.0

  # First establish nominal tracking. Reintroduce DR after closed-loop evaluation.
  cfg.events.clear()
  cfg.observations["actor"].enable_corruption = False
  for group in ("actor", "critic"):
    cfg.observations[group].terms["limbo_phase_contacts"] = ObservationTermCfg(
      func=limbo.phase_and_contacts
    )

  # Hands, knees and torso must not provide support during traditional limbo.
  cfg.scene.sensors = (
    ContactSensorCfg(name="contact_forces",
      primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=limbo.FEET),
      fields=("force",), reduce="netforce", history_length=4),
    *(ContactSensorCfg(name=f"limbo_{side}_contact",
      primary=ContactMatch(mode="body", pattern=f"{side}_ankle_roll_link", entity="robot"),
      fields=("force",), reduce="netforce", history_length=4)
      for side in ("left", "right")),
  )
  cfg.rewards["undesired_contacts"].weight = -1.0
  cfg.rewards["limbo_contact_plan"] = RewardTermCfg(func=limbo.contact_mismatch, weight=-0.5)
  cfg.rewards["limbo_stance_slip"] = RewardTermCfg(func=limbo.stance_motion, weight=-2.0)
  cfg.rewards["limbo_stance_rotation"] = RewardTermCfg(
    func=limbo.stance_motion, weight=-0.05, params={"angular": True})
  cfg.rewards["limbo_foot_position"] = RewardTermCfg(func=limbo.foot_position_tracking, weight=1.0)
  cfg.rewards["limbo_foot_orientation"] = RewardTermCfg(func=limbo.foot_orientation_tracking, weight=0.5)
  cfg.terminations["reference_end"] = TerminationTermCfg(func=limbo.reference_end, time_out=True)
  return cfg


def k1_rev1_limbo_runner_cfg():
  cfg = k1_rev1_mimic_ppo_runner_cfg()
  cfg.experiment_name = "k1_limbo_contact_v2"
  cfg.run_name = "nominal"
  cfg.algorithm.learning_rate = 3e-4
  cfg.actor.distribution_cfg["init_std"] = 0.5
  return cfg
