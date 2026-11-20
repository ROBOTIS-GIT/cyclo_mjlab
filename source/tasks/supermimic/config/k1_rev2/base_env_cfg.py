from ..k1_rev1.base_env_cfg import TRACKED_BODIES as TRACKED_BODIES, make_env_cfg as make_rev1_env_cfg


def make_env_cfg(representation="gencoord", play=False):
    from source.assets.robots.ai_sapiens.k1_rev2 import K1_REV2_FOOT_GEOMS, get_k1_rev2_cfg

    cfg = make_rev1_env_cfg(representation, play=play)
    cfg.scene.entities["robot"] = get_k1_rev2_cfg()
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = K1_REV2_FOOT_GEOMS
    return cfg
