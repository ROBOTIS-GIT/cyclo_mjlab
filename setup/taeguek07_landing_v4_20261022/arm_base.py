"""Taegeuk 7 arm stabilization on the evaluated TargetedV3 checkpoint."""
from pathlib import Path
import hashlib
import torch
import torch.nn.functional as F
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
import targeted_base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARKER = 'taeguek07_arm_smooth_v1'
SOURCE = HERE / 'source/model_9999.pt'
SOURCE_SHA = '67c2f77033b43823e132f6ba136b11023d413eb38934e7178ec1ffcdb95b4704'
TOTAL_UPDATES = 10000
MOTION = ROOT / 'source/assets/motions/K1_rev1/taeguek_07/k1_taeguek07_arm_corrected_v1_50hz_converted.npz'
MOTION_SHA = '877390d22ab136a8fc99726e240c09e73efdd2d2b9ff4d4524a6c1c90c0346df'


def validate_motion():
    if hashlib.sha256(MOTION.read_bytes()).hexdigest() != MOTION_SHA:
        raise ValueError('Motion differs from the selected Taegeuk 7 arm-corrected V1 reference.')


def revision():
    # Reject silent resumption under changed reward or entry-state definitions.
    h = hashlib.sha256()
    for name in ('refinement.py', 'targeted_base.py', 'entry_t7.npz'):
        h.update(name.encode())
        h.update((HERE / name).read_bytes())
    return h.hexdigest()


def robust_cost(x):
    return torch.sqrt(1 + x.square()) - 1


def arm_indices(cmd):
    ids = [i for i, name in enumerate(cmd.robot.joint_names)
           if any(part in name for part in ('shoulder', 'elbow', 'wrist'))]
    if len(ids) != 10:
        raise ValueError(f'Expected ten arm joints, got {ids}')
    return ids


def hold_gate(reference_velocity, fps):
    """Reference-only gate; +/-100 ms excludes momentary stops during fast moves."""
    radius = max(1, round(.1 * fps))
    speed = F.pad(reference_velocity.abs().T[None], (radius, radius), mode='replicate')
    speed = F.max_pool1d(speed, 2 * radius + 1, stride=1)[0].T
    return 1 / (1 + (speed / .35).pow(4))


class ArmStability:
    """Reference residual velocity/acceleration plus stronger near-hold damping.

    This is a temporal penalty, not a frequency-band detector. Histories are
    invalidated at resets/frame jumps; repeated reward calls do not advance them.
    """
    def __init__(self, cfg, env):
        self.cmd = env.command_manager.get_term('reference_trajectory')
        self.arms = arm_indices(self.cmd)
        reference = self.cmd.reference
        self.hold = hold_gate(reference.joint_velocity[:, self.arms], float(reference.fps))
        self.prev_frame = torch.full_like(self.cmd.frame_ids, -1)
        self.prev_error = torch.zeros_like(self.cmd.robot_joint_vel[:, self.arms])
        self.prev_ref = torch.zeros_like(self.prev_error)
        self.age = torch.zeros(env.num_envs, device=env.device)
        self.cached = torch.zeros_like(self.age)
        self.last_step = -1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.prev_frame[ids] = -1
        self.prev_error[ids] = 0
        self.prev_ref[ids] = 0
        self.age[ids] = 0
        self.cached[ids] = 0
        # Keep the global step cache: other environments must not advance twice.

    def __call__(self, env, velocity_weight=.25, acceleration_weight=.12,
                 hold_weight=.45, ramp_steps=24000):
        if self.last_step == env.common_step_counter:
            return self.cached
        frame = self.cmd.frame_ids
        continuous = (frame == self.prev_frame + 1) & (env.episode_length_buf > 1)
        self.age = torch.where(continuous, self.age + env.step_dt, 0.)
        valid = (continuous & (self.age > .15)).float()
        ref = self.cmd.joint_vel[:, self.arms]
        error = self.cmd.robot_joint_vel[:, self.arms] - ref
        acceleration = (error - self.prev_error) / env.step_dt
        ref_acceleration = (ref - self.prev_ref).abs() / env.step_dt
        costs = {
            'velocity': robust_cost(error / (1.5 + .25 * ref.abs())).mean(-1),
            'acceleration': robust_cost(acceleration / (30. + .5 * ref_acceleration)).mean(-1),
            'hold': (self.hold[frame] * robust_cost(error / .6)).mean(-1),
        }
        ramp = min(1., .1 + .9 * env.common_step_counter / max(1, ramp_steps))
        for name, cost in costs.items():
            self.cmd.metrics['arm_v1_' + name] = cost * valid
        self.cmd.metrics['arm_v1_hold_gate'] = self.hold[frame].mean(-1)
        self.cmd.metrics['arm_v1_ramp'] = torch.full_like(valid, ramp)
        self.cached = ramp * valid * (velocity_weight * costs['velocity']
            + acceleration_weight * costs['acceleration'] + hold_weight * costs['hold'])
        self.prev_frame.copy_(frame)
        self.prev_error.copy_(error)
        self.prev_ref.copy_(ref)
        self.last_step = env.common_step_counter
        return self.cached


def arm_pose_tracking(env, std=.35):
    cmd = env.command_manager.get_term('reference_trajectory')
    arms = arm_indices(cmd)
    error = cmd.robot_joint_pos[:, arms] - cmd.joint_pos[:, arms]
    error = torch.atan2(torch.sin(error), torch.cos(error))
    return torch.exp(-error.square() / std**2).mean(-1)


def env_cfg(play=False):
    validate_motion()
    cfg = targeted_base.env_cfg(7, play)
    cfg.commands['reference_trajectory'].trajectory_file = str(MOTION)
    # Retain the source policy's mature support objective from the first step.
    cfg.rewards['taeguek_support_quality'].params = dict(ramp_steps=1)
    cfg.rewards['arm_stability'] = RewardTermCfg(func=ArmStability, weight=-1., params={})
    cfg.rewards['arm_pose_tracking'] = RewardTermCfg(func=arm_pose_tracking, weight=.4,
                                                   params={'std': .35})
    return cfg


def agent_cfg():
    cfg = targeted_base.agent_cfg()
    cfg.algorithm.learning_rate = 5e-5
    cfg.algorithm.schedule = 'fixed'
    cfg.save_interval = 250
    cfg.max_iterations = TOTAL_UPDATES
    cfg.run_name = 'taeguek07_arm_smooth_v1_finetune'
    return cfg


class ArmRunner(MjlabOnPolicyRunner):
    def load(self, path, load_cfg=None, strict=True, map_location=None):
        data = torch.load(path, map_location='cpu', weights_only=False)
        marker = (data.get('infos') or {}).get(MARKER)
        if load_cfg is not None:
            raise ValueError('This task manages warm-start and resume loading explicitly.')
        if marker is not None:
            if marker.get('sha256') != SOURCE_SHA or marker.get('revision') != revision():
                raise ValueError('Resume checkpoint source/reward revision mismatch.')
            result = super().load(path, None, strict, map_location)
            completed = int(data['iter']) + 1
            if self.env.unwrapped.common_step_counter != completed * self.cfg['num_steps_per_env']:
                raise ValueError('Resume step counter inconsistent with completed updates.')
            self.current_learning_iteration = completed
            self.provenance = marker
            print('ARM_V1_RESUME', completed, 'optimizer restored', flush=True)
            return result
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if sha != SOURCE_SHA:
            raise ValueError('Warm-start requires the evaluated complete 7 model_9999.pt.')
        if self.alg.optimizer.state:
            raise RuntimeError('Warm-start requires a fresh optimizer.')
        result = super().load(path, dict(actor=True, critic=True, optimizer=False,
            iteration=False, rnd=False), strict, map_location)
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        for model, key in ((self.alg.actor, 'actor_state_dict'), (self.alg.critic, 'critic_state_dict')):
            for name, value in data[key].items():
                if not torch.equal(model.state_dict()[name].cpu(), value):
                    raise RuntimeError(f'Warm-start mismatch: {key}/{name}')
        if self.alg.optimizer.state or any(g['lr'] != self.alg.learning_rate for g in self.alg.optimizer.param_groups):
            raise RuntimeError('Source optimizer leaked into fine-tuning.')
        self.provenance = dict(source=str(Path(path).resolve()), sha256=sha,
            source_iteration=int(data['iter']), revision=revision(), fresh_optimizer=True,
            learning_rate=self.alg.learning_rate)
        print('ARM_V1_WARMSTART', self.provenance, flush=True)
        return result

    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), MARKER: self.provenance})
