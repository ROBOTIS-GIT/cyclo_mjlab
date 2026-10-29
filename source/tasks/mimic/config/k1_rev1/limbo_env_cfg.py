"""120cm limbo motion with the unchanged K1 Rev.1 Mimic defaults."""

from mjlab.envs import ManagerBasedRlEnvCfg

from source import SRC_PATH
from source.tasks.mimic.mdp import ReferenceTrajectoryCommandCfg

from .base_env_cfg import k1_rev1_mimic_env_cfg

TRAJECTORY_FILE = (
  SRC_PATH / "assets/motions/K1_rev1/limbo_1p2_contact_v2/limbo.npz"
)


def k1_rev1_limbo_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Select the limbo reference; use the standard model, rewards and reset."""
  cfg = k1_rev1_mimic_env_cfg(play=play)
  reference_trajectory = cfg.commands["reference_trajectory"]
  assert isinstance(reference_trajectory, ReferenceTrajectoryCommandCfg)
  reference_trajectory.trajectory_file = str(TRAJECTORY_FILE)
  return cfg
