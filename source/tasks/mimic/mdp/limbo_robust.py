"""Robust entry training with the existing 124-channel actor contract.

The entry blend models robotis_mujoco; it is not a deployment-side action filter.
Reference time and last_action deliberately retain the deployed semantics.
"""
from dataclasses import dataclass
import math
import numpy as np
import mujoco
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from .limbo_prepare_obs124 import Obs124PreparationCommand, Obs124PreparationCommandCfg
from .limbo_prepare_v2 import command


@requires_model_fields('actuator_gainprm', 'actuator_biasprm')
def entry_model_fields(env, env_ids):
    """Allocate independent gain arrays before the action term is constructed."""


class RobustEntryAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.capture = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.start = torch.zeros_like(self._raw_actions)
        self.duration = torch.zeros(self.num_envs, device=self.device)
        self.start_gain = torch.ones(self.num_envs, device=self.device)
        self.previous_goal = torch.zeros_like(self._raw_actions)
        self.goal_delta = torch.zeros_like(self._raw_actions)
        self.applied_target = torch.zeros_like(self._raw_actions)
        m = env.sim.mj_model
        ids = []
        for name in self.target_names:
            joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'robot/' + name)
            matches = np.flatnonzero(m.actuator_trnid[:, 0] == joint)
            if joint < 0 or len(matches) != 1:
                raise ValueError(f'Expected one native position actuator for {name}')
            ids.append(int(matches[0]))
        self.actuator_ids = torch.tensor(ids, device=self.device)
        self.kp = torch.tensor(m.actuator_gainprm[ids, 0].copy(), device=self.device, dtype=torch.float32)
        self.kd = torch.tensor(-m.actuator_biasprm[ids, 2].copy(), device=self.device, dtype=torch.float32)
        self.effort = torch.tensor(np.max(np.abs(m.actuator_forcerange[ids]), axis=-1), device=self.device, dtype=torch.float32)
        if not (self.kp > 0).all() or not (self.effort > 0).all():
            raise ValueError('Entry model requires native position PD and finite torque limits')

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.capture[ids] = True
        self.elapsed[ids] = 0

    def process_actions(self, actions):
        super().process_actions(actions)
        ids = self.capture.nonzero().flatten()
        c = command(self.env)
        if len(ids):
            self.start[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]
            self.previous_goal[ids] = self.start[ids]
            self.duration[ids] = torch.where(c.prep_origin[ids], c.entry_duration[ids], 0.)
            self.start_gain[ids] = c.entry_start_gain[ids]
            self.capture[ids] = False
        self.goal_delta[:] = self._processed_actions - self.previous_goal
        self.previous_goal[:] = self._processed_actions

    def apply_actions(self):
        alpha = torch.where(self.duration > 0,
            .5 * (1 - torch.cos(math.pi * (self.elapsed / self.duration.clamp_min(1e-8)).clamp(0, 1))),
            torch.ones_like(self.elapsed))
        self.applied_target[:] = self.start + alpha[:, None] * (self._processed_actions - self.start)
        gain = self.start_gain + alpha * (1 - self.start_gain)
        model = self.env.sim.model
        model.actuator_gainprm[:, self.actuator_ids, 0] = gain[:, None] * self.kp
        model.actuator_biasprm[:, self.actuator_ids, 1] = -gain[:, None] * self.kp
        model.actuator_biasprm[:, self.actuator_ids, 2] = -gain[:, None] * self.kd
        self._entity.set_joint_position_target(
            self.applied_target - self._entity.data.encoder_bias[:, self._target_ids], joint_ids=self._target_ids)
        self.elapsed += self.env.physics_dt


@dataclass(kw_only=True)
class RobustEntryActionCfg(JointPositionActionCfg):
    def build(self, env):
        return RobustEntryAction(self, env)


class RobustPreparationCommand(Obs124PreparationCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, device = self.num_envs, self.device
        self.entry_duration = torch.ones(n, device=device)
        self.entry_start_gain = torch.ones(n, device=device)
        self.push_at = torch.full((n,), float('inf'), device=device)
        self.push_delta = torch.zeros((n, 6), device=device)
        self.push_done = torch.zeros(n, dtype=torch.bool, device=device)
        self.reset_level = torch.zeros(n, device=device)
        self._build_reset_pool()
        for k in ('entry_curriculum', 'entry_duration', 'entry_push_count'):
            self.metrics[k] = torch.zeros(n, device=device)

    def difficulty(self):
        return min(1., max(self.cfg.initial_difficulty,
            float(self._env.common_step_counter) / self.cfg.robust_curriculum_steps))

    def _build_reset_pool(self):
        """FK-project reproducible reset poses; no deep initial foot penetration.

        This runs once on the CPU, not on every reset. Joint limits and foot
        sphere geometry are taken from the actual compiled training robot.
        """
        m = self._env.sim.mj_model
        d = mujoco.MjData(m)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'robot/' + n) for n in self.robot.joint_names]
        qadr = m.jnt_qposadr[jids]
        limits = m.jnt_range[jids]
        feet = [i for i in range(m.ngeom) if 'ankle_roll_link_collision_' in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '')]
        if len(feet) != 18 or not np.all(m.geom_type[feet] == mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError('Reset projection expects K1 Rev1 eighteen foot spheres')
        rng = np.random.default_rng(self.cfg.reset_pool_seed)
        base_r = self.reset_roots.cpu().numpy()
        base_j = self.reset_joints.cpu().numpy()
        roots, joints = [], []
        for level in (.2, .5, 1.):
            rr, jj = [], []
            for k in range(self.cfg.reset_pool_size):
                ix = 64 if k % 2 == 0 else int(rng.integers(len(base_r)))
                root, joint = base_r[ix].copy(), base_j[ix].copy()
                # Include the actual AMP neutral arm/leg configuration as well.
                if k % 4 == 0:
                    joint = np.array([-.312,0,0,.669,-.363,0,-.312,0,0,.669,-.363,0,0,.2,.2,0,.6,0,.2,-.2,0,.6,0])
                amplitude = np.array([.035 if i < 13 else .10 for i in range(23)]) * level
                joint += rng.uniform(-amplitude, amplitude)
                joint = np.clip(joint, limits[:, 0] + .003, limits[:, 1] - .003)
                roll, pitch = rng.uniform(-.07, .07, 2) * level
                delta = np.array([math.cos(roll/2)*math.cos(pitch/2), math.sin(roll/2)*math.cos(pitch/2),
                                  math.cos(roll/2)*math.sin(pitch/2), -math.sin(roll/2)*math.sin(pitch/2)])
                quat = np.zeros(4); mujoco.mju_mulQuat(quat, root[3:], delta); root[3:] = quat
                root[:2] += rng.uniform(-.015, .015, 2) * level
                d.qpos[:7] = root; d.qpos[qadr] = joint
                mujoco.mj_forward(m, d)
                bottom = np.min(d.geom_xpos[feet, 2] - m.geom_size[feet, 0])
                root[2] += .001 - bottom + rng.uniform(0, .004) * level
                rr.append(root); jj.append(joint)
            roots.append(rr); joints.append(jj)
        self.pool_roots = torch.tensor(np.asarray(roots), device=self.device, dtype=torch.float32)
        self.pool_joints = torch.tensor(np.asarray(joints), device=self.device, dtype=torch.float32)

    def _sample_start_frames(self, env_ids):
        super()._sample_start_frames(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        replay = ids[~self.prep_origin[ids]]
        self.frame_ids[replay] = torch.randint(self.reference.num_frames, (len(replay),), device=self.device)
        self.started_at_zero[ids] = self.frame_ids[ids] == 0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        prep = ids[self.prep_origin[ids]]
        level = self.cfg.evaluation_difficulty if self.cfg.start_from_zero else self.difficulty()
        self.reset_level[ids] = level
        self.entry_duration[ids] = 1.
        self.entry_start_gain[ids] = 1.
        self.push_done[ids] = False
        self.push_at[ids] = float('inf')
        if not self.cfg.start_from_zero:
            # Full one-second deployment entry is present from the first update.
            u = torch.rand(len(ids), device=self.device)
            direct_fraction = .35 - .20 * level
            short_fraction = .55 - .40 * level
            self.entry_duration[ids] = torch.where(u < direct_fraction, 0.,
                torch.where(u < direct_fraction + short_fraction, .15 + .35 * level, 1.))
            self.entry_start_gain[ids] = torch.where(torch.rand(len(ids), device=self.device) < .25,
                1. + (-.4 + .5 * torch.rand(len(ids), device=self.device)) * level, 1.)
        if len(prep) and self.cfg.randomize_entry:
            # Keep nominal reset-bank examples while widening the recovery basin.
            # Evaluation always uses the disjoint perturbed pool for every case.
            prep = prep if self.cfg.start_from_zero else prep[torch.rand(len(prep),device=self.device) < .25 + .5 * level]
            pool = 0 if level <= .2 else 1 if level <= .5 else 2
            indices = torch.randint(self.cfg.reset_pool_size, (len(prep),), device=self.device)
            roots = self.pool_roots[pool, indices].clone()
            roots[:, :3] += self._env.scene.env_origins[prep]
            joints = self.pool_joints[pool, indices]
            joint_vel = (torch.rand_like(joints)*2-1) * (.35 * level)
            velocity = (torch.rand((len(prep),6),device=self.device)*2-1) * torch.tensor([.3,.25,.03,.4,.4,.3],device=self.device) * level
            self.robot.write_joint_state_to_sim(joints, joint_vel, env_ids=prep)
            self.robot.write_root_link_pose_to_sim(roots, env_ids=prep)
            self.robot.write_root_link_velocity_to_sim(velocity, env_ids=prep)
            self.robot.clear_state(env_ids=prep)
        prep_ids = ids[self.prep_origin[ids]]
        if self.cfg.enable_entry_push and len(prep_ids):
            chosen = prep_ids[torch.rand(len(prep_ids),device=self.device) < self.cfg.push_probability]
            self.push_at[chosen] = .2 + 1.3*torch.rand(len(chosen),device=self.device)
            self.push_delta[chosen] = (torch.rand((len(chosen),6),device=self.device)*2-1) * torch.tensor([.2,.2,0,.2,.2,.15],device=self.device) * level

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics['entry_curriculum'][:] = self.reset_level
        self.metrics['entry_duration'][:] = self.entry_duration
        self.metrics['entry_push_count'][:] = self.push_done.float()


@dataclass(kw_only=True)
class RobustPreparationCommandCfg(Obs124PreparationCommandCfg):
    randomize_entry: bool = True
    enable_entry_push: bool = True
    push_probability: float = .65
    initial_difficulty: float = .2
    robust_curriculum_steps: int = 144000  # 6000 PPO iterations * 24 steps.
    evaluation_difficulty: float = 1.
    reset_pool_seed: int = 20261028
    reset_pool_size: int = 256
    def build(self, env):
        return RobustPreparationCommand(self, env)


@requires_model_fields('geom_friction')
def curriculum_material(env, env_ids, asset_cfg):
    from source.tasks.mimic import mdp
    c = command(env)
    level = c.cfg.evaluation_difficulty if c.cfg.start_from_zero else c.difficulty()
    mdp.randomize_rigid_body_material(env, env_ids, asset_cfg=asset_cfg,
        ranges=(.6-.1*level, .6+.5*level), operation='abs', shared_random=True)


def entry_push(env, env_ids=None):
    c = command(env)
    elapsed = env.episode_length_buf * env.step_dt
    ids = (c.prep_origin & ~c.push_done & (elapsed >= c.push_at) & (elapsed < 1.7)).nonzero().flatten()
    if len(ids):
        velocity = c.robot.data.root_link_vel_w[ids].clone() + c.push_delta[ids]
        c.robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
        c.push_done[ids] = True


def entry_action_cost(env, kind):
    c = command(env)
    a = env.action_manager.get_term('joint_pos')
    active = c.prep_origin & (env.episode_length_buf * env.step_dt <= 2.)
    if kind == 'excess':
        value = torch.relu(a.raw_action.abs() - 2.).square().mean(-1)
    elif kind == 'jump':
        value = a.goal_delta.square().mean(-1) / .25**2
    elif kind == 'saturation':
        ratio = env.sim.data.actuator_force[:, a.actuator_ids].abs() / a.effort
        value = torch.relu(ratio - .8).square().mean(-1)
    else:
        raise ValueError(kind)
    return value.clamp(max=100) * active
