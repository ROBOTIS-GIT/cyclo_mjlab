"""Isolated Taegeuk 5 reward tuning; observations, actions and reference unchanged."""
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.config.k1_rev1.taeguek_support_v2_env_cfg import (
    k1_rev1_taeguek_support_v2_env_cfg, k1_rev1_taeguek_support_v2_ppo_cfg,
)
from source.tasks.mimic.mdp.taeguek_support_v2 import TaegeukFootSupportV2, sole_bottom_height



def robust_cost(x):
    """Quadratic near zero, linear tails, capped recovery cost."""
    return (torch.sqrt(1 + x.square()) - 1).clamp_max(4)


def moving_clearance(actual, target, speed, eligible):
    # Only reference speed controls eligibility/tolerance; speeding the robot up
    # cannot evade this cost. Fast relocation and kicks are excluded upstream.
    tolerance = .003 + .004 * speed.clamp(0, .8)
    excess = (actual - target.clamp_min(0) - tolerance).clamp_min(0)
    gate = (1 - speed / .8).clamp(0, 1) * eligible
    return robust_cost(excess / .010) * gate


class MovingFootSupport(TaegeukFootSupportV2):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Require a low reference sole throughout +/-40 ms, protecting real
        # liftoff/landing transitions. Do not overlap existing support masks.
        low = self.reference_bottom <= .012
        padded = torch.nn.functional.pad(low.T.float()[None], (2, 2), mode='replicate')
        stable_low = (-torch.nn.functional.max_pool1d(-padded, 5, stride=1))[0].T > .5
        self.moving_eligible = stable_low & ~self.reference_support & ~self.reference_kick

    def __call__(self, env, moving_weight=.35, new_ramp_steps=24000):
        # Preserve the mature selected V2 objective from the first physical step.
        base = super().__call__(env, refinement_ramp_steps=1, ramp_steps=1)
        frame = self.cmd.frame_ids
        bottom = sole_bottom_height(self.cmd.robot_body_pos_w[:, self.feet],
            self.cmd.robot_body_quat_w[:, self.feet], self.offsets, self.radii)
        bottom = bottom - env.scene.env_origins[:, 2, None]
        cost = moving_clearance(bottom, self.reference_bottom[frame],
            self.reference_lowpoint_speed[frame], self.moving_eligible[frame]).sum(-1)
        valid = (self.age > .15).float()
        ramp = min(1., .1 + .9 * env.common_step_counter / new_ramp_steps)
        self.cmd.metrics['smooth_v3_moving_clearance'] = cost * valid
        return base + moving_weight * ramp * cost * valid


class ReferenceResidualSmoothness:
    """Penalize changes in tracking error, not the requested punch/kick itself.

    Finite differences use consecutive reference frames only, after a reset
    grace period. All caches are per environment; repeat manager calls are safe.
    These are broad temporal smoothness terms, not a 4--15 Hz spectral filter.
    """
    def __init__(self, cfg, env):
        self.cmd = env.command_manager.get_term('reference_trajectory')
        self.arms = [i for i, name in enumerate(self.cmd.robot.joint_names)
                     if any(part in name for part in ('shoulder', 'elbow', 'wrist'))]
        if len(self.arms) != 10:
            raise ValueError(f'Expected 10 arm joints, got {self.arms}')
        self.torso = self.cmd.cfg.body_names.index('torso_link')
        self.prev_frame = torch.full_like(self.cmd.frame_ids, -1)
        self.prev_joint = torch.zeros_like(self.cmd.robot_joint_vel)
        self.prev_reference = torch.zeros_like(self.prev_joint)
        self.prev_torso = torch.zeros((env.num_envs, 2), device=env.device)
        self.age = torch.zeros(env.num_envs, device=env.device)
        self.cached = torch.zeros_like(self.age)
        self.last_step = -1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.prev_frame[ids] = -1
        self.prev_joint[ids] = 0
        self.prev_reference[ids] = 0
        self.prev_torso[ids] = 0
        self.age[ids] = 0
        self.cached[ids] = 0
        # Keep same-step cache valid for environments not being reset.

    def __call__(self, env, torso_velocity_weight=.35, torso_acceleration_weight=.12,
                 arm_velocity_weight=.20, joint_acceleration_weight=.08,
                 arm_acceleration_weight=.08, ramp_steps=24000):
        if self.last_step == env.common_step_counter:
            return self.cached
        frame = self.cmd.frame_ids
        continuous = (frame == self.prev_frame + 1) & (env.episode_length_buf > 1)
        self.age = torch.where(continuous, self.age + env.step_dt, 0.)
        valid = (continuous & (self.age > .15)).float()
        ref = self.cmd.joint_vel
        joint = self.cmd.robot_joint_vel - ref
        # World horizontal angular velocity: leave desired world-yaw turns to
        # the original orientation/velocity tracking terms.
        torso = (self.cmd.robot_body_ang_vel_w[:, self.torso, :2]
                 - self.cmd.body_ang_vel_w[:, self.torso, :2])
        joint_acc = (joint - self.prev_joint) / env.step_dt
        ref_acc = (ref - self.prev_reference).abs() / env.step_dt
        torso_acc = (torso - self.prev_torso) / env.step_dt
        costs = {
            'torso_velocity': robust_cost(torso / .6).mean(-1),
            'torso_acceleration': robust_cost(torso_acc / 15.).mean(-1),
            'arm_velocity': robust_cost(joint[:, self.arms] / (1.5 + .25 * ref[:, self.arms].abs())).mean(-1),
            'joint_acceleration': robust_cost(joint_acc / (40. + .5 * ref_acc)).mean(-1),
            'arm_acceleration': robust_cost(joint_acc[:, self.arms] / (30. + .5 * ref_acc[:, self.arms])).mean(-1),
        }
        weights = (torso_velocity_weight, torso_acceleration_weight,
                   arm_velocity_weight, joint_acceleration_weight, arm_acceleration_weight)
        ramp = min(1., .1 + .9 * env.common_step_counter / ramp_steps)
        for name, cost in costs.items():
            self.cmd.metrics['smooth_v3_' + name] = cost * valid
        self.cmd.metrics['smooth_v3_ramp'] = torch.full_like(valid, ramp)
        self.cached = ramp * valid * sum(w * cost for w, cost in zip(weights, costs.values()))
        self.prev_frame.copy_(frame)
        self.prev_joint.copy_(joint)
        self.prev_reference.copy_(ref)
        self.prev_torso.copy_(torso)
        self.last_step = env.common_step_counter
        return self.cached


def arm_pose_tracking(env, std=.45):
    cmd = env.command_manager.get_term('reference_trajectory')
    arms = [i for i, name in enumerate(cmd.robot.joint_names)
            if any(part in name for part in ('shoulder', 'elbow', 'wrist'))]
    error = cmd.robot_joint_pos[:, arms] - cmd.joint_pos[:, arms]
    error = torch.atan2(torch.sin(error), torch.cos(error))
    return torch.exp(-error.square() / std**2).mean(-1)


def env_cfg(play=False):
    cfg = k1_rev1_taeguek_support_v2_env_cfg(5, play)
    term = cfg.rewards['taeguek_support_quality']
    term.func = MovingFootSupport
    term.params = dict(moving_weight=.35, new_ramp_steps=24000)
    cfg.rewards['reference_residual_smoothness'] = RewardTermCfg(
        func=ReferenceResidualSmoothness, weight=-1., params={})
    cfg.rewards['arm_pose_tracking'] = RewardTermCfg(func=arm_pose_tracking, weight=.35, params={'std': .45})
    if not play:
        # Full 40.32 s sequence fits; retain adaptive resets and disturbances.
        cfg.episode_length_s = 42.34
        cmd = cfg.commands['reference_trajectory']
        cmd.first_frame_env_fraction = .3
        cmd.start_sampling_uniform_mix = .2
    return cfg


def agent_cfg():
    cfg = k1_rev1_taeguek_support_v2_ppo_cfg()
    cfg.algorithm.learning_rate = 5e-5
    cfg.save_interval = 250
    cfg.upload_model = False
    cfg.run_name = 'taeguek05_smooth_v3_finetune'
    return cfg
