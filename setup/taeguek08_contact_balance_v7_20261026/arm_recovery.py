import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul, quat_error_magnitude

def robust_cost(error):
    # No upper clipping: even a large pose error still distinguishes recovery.
    return torch.sqrt(1 + error.square()) - 1


def hold_gate(reference_velocity, fps):
    radius = max(1, round(.1 * fps))
    speed = F.pad(reference_velocity.abs().T[None], (radius, radius), mode='replicate')
    speed = F.max_pool1d(speed, 2 * radius + 1, stride=1)[0].T
    return 1 / (1 + (speed / .35).pow(4))


def relative_pose(positions, orientations, torso, bodies):
    """Each skeleton's own torso frame; no penalty for global placement/yaw."""
    origin = positions[:, torso:torso + 1]
    rotation = orientations[:, torso:torso + 1].expand(-1, len(bodies), -1)
    return (quat_apply_inverse(rotation, positions[:, bodies] - origin),
            quat_mul(quat_conjugate(rotation), orientations[:, bodies]))


class ArmRecovery:
    """Dense large-error recovery plus small, reference-gated hold damping."""
    def __init__(self, cfg, env):
        self.cmd = env.command_manager.get_term('reference_trajectory')
        names = self.cmd.robot.joint_names
        self.arms = [i for i, name in enumerate(names)
                     if any(part in name for part in ('shoulder', 'elbow', 'wrist'))]
        if len(self.arms) != 10:
            raise ValueError('Expected ten arm joints')
        # V4 regressed most in elbows/wrists; preserve these as well as shoulders.
        weights = [2.0 if 'wrist' in names[i] else (1.5 if 'elbow' in names[i] else 1.) for i in self.arms]
        self.joint_weights = torch.tensor(weights, device=env.device)
        self.joint_weights /= self.joint_weights.mean()
        bodies = self.cmd.cfg.body_names
        self.torso = bodies.index('torso_link')
        self.bodies = [bodies.index(name) for name in (
            'left_elbow_link', 'right_elbow_link',
            'left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand')]
        reference = self.cmd.reference
        self.hold = hold_gate(reference.joint_velocity[:, self.arms], float(reference.fps))
        self.prev_frame = torch.full_like(self.cmd.frame_ids, -1)
        self.prev_velocity = torch.zeros_like(self.cmd.robot_joint_vel[:, self.arms])
        self.age = torch.zeros(env.num_envs, device=env.device)
        self.cached = torch.zeros_like(self.age)
        self.last_step = -1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.prev_frame[ids] = -1
        self.prev_velocity[ids] = 0
        self.age[ids] = 0
        self.cached[ids] = 0

    def __call__(self, env, joint_weight=.60, elbow_weight=.25, hand_weight=.50,
                 orientation_weight=.20, hold_velocity_weight=.18,
                 hold_acceleration_weight=.04, ramp_steps=1):
        if self.last_step == env.common_step_counter:
            return self.cached
        cmd = self.cmd
        frame = cmd.frame_ids
        continuous = (frame == self.prev_frame + 1) & (env.episode_length_buf > 1)
        self.age = torch.where(continuous, self.age + env.step_dt, 0.)
        valid = (continuous & (self.age > .15)).float()
        pose_error = cmd.robot_joint_pos[:, self.arms] - cmd.joint_pos[:, self.arms]
        # These revolute joints have physical limits: a 2*pi displacement must
        # not receive a perfect pose reward through angle wrapping.
        joint_cost = (robust_cost(pose_error / .35) * self.joint_weights).mean(-1)
        actual_pos, actual_rot = relative_pose(cmd.robot_body_pos_w, cmd.robot_body_quat_w,
                                               self.torso, self.bodies)
        target_pos, target_rot = relative_pose(cmd.body_pos_w, cmd.body_quat_w,
                                               self.torso, self.bodies)
        position_error = torch.linalg.vector_norm(actual_pos - target_pos, dim=-1)
        orientation_error = quat_error_magnitude(actual_rot[:, 2:], target_rot[:, 2:])
        velocity_error = cmd.robot_joint_vel[:, self.arms] - cmd.joint_vel[:, self.arms]
        acceleration_error = (velocity_error - self.prev_velocity) / env.step_dt
        # Reference-only temporal gate protects intentional fast motion. Extra
        # hold damping fades when far from pose, permitting corrective movement;
        # the recovery costs remain active and cannot be evaded by staying wrong.
        near_pose = 1 / (1 + (pose_error / .35).pow(4))
        gate = self.hold[frame] * near_pose
        costs = {
            'joint': joint_cost,
            'elbow': robust_cost(position_error[:, :2] / .06).mean(-1),
            'hand': robust_cost(position_error[:, 2:] / .08).mean(-1),
            'orientation': robust_cost(orientation_error / .5).mean(-1),
            'hold_velocity': (gate * robust_cost(velocity_error / .6)).mean(-1) * valid,
            'hold_acceleration': (gate * robust_cost(acceleration_error / 20.)).mean(-1) * valid,
        }
        weights = (joint_weight, elbow_weight, hand_weight, orientation_weight,
                   hold_velocity_weight, hold_acceleration_weight)
        ramp = min(1., .1 + .9 * env.common_step_counter / max(1, ramp_steps))
        self.cached = ramp * sum(w * v for w, v in zip(weights, costs.values()))
        for name, value in costs.items():
            cmd.metrics['t8_arm_'  + name] = value
        cmd.metrics['t8_arm_rmse_rad'] = pose_error.square().mean(-1).sqrt()
        cmd.metrics['t8_arm_joint_error_rad'] = pose_error.abs().mean(-1)
        cmd.metrics['t8_arm_hand_error_m'] = position_error[:, 2:].mean(-1)
        cmd.metrics['t8_arm_hold_gate'] = gate.mean(-1)
        cmd.metrics['t8_arm_ramp'] = torch.full_like(self.age, ramp)
        self.prev_frame.copy_(frame)
        self.prev_velocity.copy_(velocity_error)
        self.last_step = env.common_step_counter
        return self.cached

