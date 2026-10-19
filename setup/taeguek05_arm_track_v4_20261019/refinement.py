"""Taegeuk 5: recover arm pose errors and stabilize reference holds."""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.lab_api.math import (
    quat_apply_inverse, quat_conjugate, quat_mul, quat_error_magnitude,
)
import smooth_v3_base as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'source/model_9999.pt'
SOURCE_SHA = '00678994ac8fff241f521e9251520f2b562507bc1e9a0a10d25755e6bc4011a3'
MOTION = ROOT / 'source/assets/motions/K1_rev1/taeguek_05/k1_taeguek05_collision_only_50hz_converted.npz'
MOTION_SHA = 'b5c0c1ece88adc2e78ab0abf239a583442cf04b0a00553c89110ec4d4649d744'
MARKER = 'taeguek05_arm_track_v4'
RUN_NAME = MARKER + '_finetune'
TOTAL_UPDATES = 10000


def revision():
    digest = hashlib.sha256()
    for name in ('refinement.py', 'smooth_v3_base.py'):
        digest.update(name.encode())
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def validate_inputs():
    for path, expected in ((SOURCE, SOURCE_SHA), (MOTION, MOTION_SHA)):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}. See README.md for checkpoint transfer.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input fingerprint mismatch: {path}')


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
        weights = [1.5 if 'shoulder' in names[i] else 1. for i in self.arms]
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
                 hold_acceleration_weight=.04, ramp_steps=24000):
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
        joint_cost = (robust_cost(pose_error / .5) * self.joint_weights).mean(-1)
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
            cmd.metrics['arm_v4_' + name] = value
        cmd.metrics['arm_v4_joint_error_rad'] = pose_error.abs().mean(-1)
        cmd.metrics['arm_v4_hand_error_m'] = position_error[:, 2:].mean(-1)
        cmd.metrics['arm_v4_hold_gate'] = gate.mean(-1)
        cmd.metrics['arm_v4_ramp'] = torch.full_like(self.age, ramp)
        self.prev_frame.copy_(frame)
        self.prev_velocity.copy_(velocity_error)
        self.last_step = env.common_step_counter
        return self.cached


def env_cfg(play=False):
    cfg = base.env_cfg(play)
    # Compose has already learned these penalties; do not reset their curriculum.
    cfg.rewards['taeguek_support_quality'].params['new_ramp_steps'] = 1
    cfg.rewards['reference_residual_smoothness'].params['ramp_steps'] = 1
    cfg.rewards['arm_recovery'] = RewardTermCfg(func=ArmRecovery, weight=-1., params={})
    return cfg


def agent_cfg():
    cfg = base.agent_cfg()
    cfg.algorithm.learning_rate = 5e-5
    cfg.algorithm.schedule = 'fixed'
    cfg.algorithm.entropy_coef = .001
    cfg.max_iterations = TOTAL_UPDATES
    cfg.run_name = RUN_NAME
    return cfg


def validate_resume(data, num_envs=None):
    meta = (data.get('infos') or {}).get(MARKER)
    if not meta or meta.get('sha256') != SOURCE_SHA or meta.get('revision') != revision():
        raise ValueError('Checkpoint source/reward revision mismatch; refusing silent restart.')
    completed = int(data['iter']) + 1
    if not 0 < completed <= TOTAL_UPDATES:
        raise ValueError('Invalid completed update count')
    if data['infos']['env_state']['common_step_counter'] != completed * 24:
        raise ValueError('Checkpoint iteration/environment counter mismatch')
    if num_envs is not None and meta.get('num_envs') != num_envs:
        raise ValueError(f'Resume with --num-envs {meta.get("num_envs")} to preserve batch size.')
    if not data.get('optimizer_state_dict', {}).get('state'):
        raise ValueError('Checkpoint has no trained optimizer state')
    if not all(torch.isfinite(v).all() for key in ('actor_state_dict', 'critic_state_dict')
               for v in data[key].values()):
        raise ValueError('Non-finite policy checkpoint')
    return completed


class ArmRunner(MjlabOnPolicyRunner):
    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None:
            raise ValueError('This task manages warm-start and resume loading.')
        map_location = map_location or self.device
        data = torch.load(path, map_location='cpu', weights_only=False)
        if (data.get('infos') or {}).get(MARKER):
            completed = validate_resume(data, self.env.unwrapped.num_envs)
            result = super().load(path, None, strict, map_location)
            self.current_learning_iteration = completed
            self.provenance = data['infos'][MARKER]
            print('ARM_V4_RESUME', completed, flush=True)
            return result
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError('Warm-start requires the evaluated Compose SmoothV3 model_9999.pt.')
        result = super().load(path, dict(actor=True, critic=True, optimizer=False,
            iteration=False, rnd=False), strict, map_location)
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        if self.alg.optimizer.state:
            raise RuntimeError('Warm-start optimizer must be fresh')
        self.provenance = dict(sha256=SOURCE_SHA, source_iteration=int(data['iter']),
            revision=revision(), num_envs=self.env.unwrapped.num_envs, fresh_optimizer=True)
        print('ARM_V4_WARMSTART', self.provenance, flush=True)
        return result

    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), MARKER: self.provenance})
