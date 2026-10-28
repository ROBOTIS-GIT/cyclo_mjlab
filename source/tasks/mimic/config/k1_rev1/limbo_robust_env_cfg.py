"""V6: BeyondMimic tracking plus robust, deployed-semantics preparation."""
from dataclasses import dataclass, fields
import copy
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from source import SRC_PATH
from source.tasks.mimic.mdp import limbo_robust as robust
from source.tasks.mimic.tracking_env_cfg import make_mimic_env_cfg
from .limbo_preserve_env_cfg import k1_rev1_limbo_preserve_env_cfg, k1_rev1_limbo_preserve_runner_cfg, PreservePpoCfg


@dataclass
class RobustPpoCfg(PreservePpoCfg):
    prep_reference_file: str = ''
    anchor_action_std: float = .2


def k1_rev1_limbo_robust_env_cfg(play=False):
    cfg = k1_rev1_limbo_preserve_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 4096
    old = cfg.commands['reference_trajectory']
    values = {f.name:getattr(old, f.name) for f in fields(old)}
    values['preparation_fraction'] = .75
    cfg.commands['reference_trajectory'] = robust.RobustPreparationCommandCfg(**values)
    old_action = cfg.actions['joint_pos']
    cfg.actions['joint_pos'] = robust.RobustEntryActionCfg(**{f.name:getattr(old_action,f.name) for f in fields(old_action)})
    cfg.events['entry_model_fields'] = EventTermCfg(func=robust.entry_model_fields, mode='startup')
    if not play:
        # Restore bounded randomization removed by the original nominal task.
        defaults = make_mimic_env_cfg().events
        material = copy.deepcopy(defaults['physics_material'])
        material.func = robust.curriculum_material
        material.mode = 'reset'
        material.params = {'asset_cfg':material.params['asset_cfg']}
        cfg.events['physics_material'] = material
        home = copy.deepcopy(defaults['joint_home_offset_noise'])
        home.params['pos_distribution_params'] = (-.003, .003)
        cfg.events['joint_home_offset_noise'] = home
        com = copy.deepcopy(defaults['torso_com_offset_noise'])
        com.params['asset_cfg'] = SceneEntityCfg('robot',body_names=('torso_link',))
        com.params['com_range'] = {'x':(-.005,.005),'y':(-.005,.005),'z':(-.005,.005)}
        cfg.events['torso_com_offset_noise'] = com
        cfg.events['entry_push'] = EventTermCfg(func=robust.entry_push,mode='step')
        cfg.observations['actor'].enable_corruption = True
        # Moderate observation noise; observation dimensions/order stay fixed.
        for term in cfg.observations['actor'].terms.values():
            if term.noise is not None:
                term.noise.n_min *= .5
                term.noise.n_max *= .5
    else:
        cfg.commands['reference_trajectory'].randomize_entry = False
        cfg.commands['reference_trajectory'].enable_entry_push = False
    # Preserve all six BeyondMimic tracking rewards and motion/contact costs.
    cfg.rewards['limbo_full_success'].weight = 100.
    cfg.rewards['prepare_reached'].weight = 100.
    cfg.rewards['prepare_failed'].weight = -100.
    cfg.rewards['prepare_orientation'].weight = -.5
    cfg.rewards['prepare_stability'].weight = -.15
    cfg.rewards['entry_action_excess'] = RewardTermCfg(func=robust.entry_action_cost,weight=-.15,params={'kind':'excess'})
    cfg.rewards['entry_target_jump'] = RewardTermCfg(func=robust.entry_action_cost,weight=-.10,params={'kind':'jump'})
    cfg.rewards['entry_torque_saturation'] = RewardTermCfg(func=robust.entry_action_cost,weight=-.5,params={'kind':'saturation'})
    return cfg


def k1_rev1_limbo_robust_runner_cfg():
    cfg = k1_rev1_limbo_preserve_runner_cfg()
    cfg.experiment_name = 'k1_limbo_1p2_robust_obs124'
    cfg.run_name = 'bar120_robust_v6'
    cfg.save_interval = 1000
    cfg.max_iterations = 20000
    cfg.actor.distribution_cfg['init_std'] = .08
    cfg.algorithm = RobustPpoCfg(**{f.name:getattr(cfg.algorithm,f.name) for f in fields(cfg.algorithm)})
    cfg.algorithm.class_name = 'source.tasks.mimic.limbo_robust_ppo:RobustPPO'
    cfg.algorithm.prep_reference_file = str(SRC_PATH/'assets/motions/K1_rev1/limbo_1p2_obs124/limbo.npz')
    cfg.algorithm.teacher_coef = .05
    cfg.algorithm.replay_coef = .1
    cfg.algorithm.learning_rate = 1e-5
    return cfg
