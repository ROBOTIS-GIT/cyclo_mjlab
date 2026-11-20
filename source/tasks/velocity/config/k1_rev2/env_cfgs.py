from mjlab.envs import ManagerBasedRlEnvCfg

from source.assets.robots.ai_sapiens.k1_rev2 import K1_REV2_FOOT_GEOMS, get_k1_rev2_cfg
from ..k1_rev1.env_cfgs import cyclo_k1_rev1_flat_env_cfg


def cyclo_k1_rev2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = cyclo_k1_rev1_flat_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_k1_rev2_cfg()
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = K1_REV2_FOOT_GEOMS
  return cfg
