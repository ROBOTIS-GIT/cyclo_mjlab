import torch
from mjlab.utils.lab_api.math import quat_apply
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height
MAX_SUPPORT_COST = .6

def support_ramp(steps, ramp_steps=120000):
    return min(1., .1 + .9 * steps / max(1, ramp_steps))

def bounded_support_cost(raw):
    return MAX_SUPPORT_COST * torch.tanh(raw / MAX_SUPPORT_COST)

def huber(x):
    return torch.sqrt(1 + x.square()) - 1


def foot_costs(drift, angular_error, angular_acc_error, bottom, target_bottom,
               vertical_speed_error, support):
    """All geometric terms stay active without contact; no hovering loophole."""
    return {
        'drift': huber((drift.norm(dim=-1) - .02).clamp_min(0) / .05) * support,
        'rocking': huber((angular_error[..., :2].norm(dim=-1) - .35).clamp_min(0)) * support,
        'rocking_acc': huber((angular_acc_error[..., :2].norm(dim=-1) - 5).clamp_min(0) / 20) * support,
        'clearance': huber((bottom - target_bottom.clamp_min(0) - .01).clamp_min(0) / .02) * support,
        'rising': huber((vertical_speed_error - .05).clamp_min(0) / .25) * support,
    }


def phase_drift(error, offset, support, previous_support, continuous):
    start = support & (~previous_support | ~continuous[:, None])
    offset = torch.where(start[..., None], error, offset)
    return error - offset, offset


class SupportStability(TaegeukFootSupportQuality):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.point = torch.tensor([.0325, 0, -.065], device=env.device)
        ref = self.cmd.reference
        self.ref_sole = ref.body_position_w[:, self.feet] + quat_apply(
            ref.body_orientation_w[:, self.feet],
            self.point.expand(ref.num_frames, 2, 3))
        self.target_vz = torch.gradient(self.reference_bottom,
                                       spacing=(1 / float(ref.fps),), dim=0)[0]
        self.phase_offset = torch.zeros(env.num_envs, 2, 2, device=env.device)
        self.prev_expected = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.prev_angular_error = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self.prev_bottom = torch.zeros(env.num_envs, 2, device=env.device)
        self.geom_frame = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.geom_cache = torch.zeros(env.num_envs, device=env.device)
        self.geom_step = -1
        print('T6_SUPPORT_V2_REFERENCE_SECONDS',
              (self.reference_support.sum(0) / float(ref.fps)).tolist(), flush=True)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, 'geom_frame'):
            ids = slice(None) if env_ids is None else env_ids
            self.geom_frame[ids] = -1
            self.prev_expected[ids] = False
            self.phase_offset[ids] = 0
            self.prev_angular_error[ids] = 0
            self.geom_cache[ids] = 0
            self.geom_step = -1

    def __call__(self, env, drift_weight=.08, rocking_weight=.05,
                 rocking_acc_weight=0., clearance_weight=.08,
                 rising_weight=.02, ramp_steps=120000):
        if self.geom_step == env.common_step_counter:
            return self.geom_cache
        # The base class handles reference discontinuities and reset grace.
        base = super().__call__(
            env, support_weight=.2, long_unload_weight=.05, liftoff_weight=.01,
            slip_weight=.1, rotation_weight=0.,
            single_support_multiplier=1.2, ramp_steps=1)
        frame = self.cmd.frame_ids
        support = self.reference_support[frame]
        continuous = (frame == self.geom_frame + 1) & (env.episode_length_buf > 1)
        position = self.cmd.robot_body_pos_w[:, self.feet]
        quaternion = self.cmd.robot_body_quat_w[:, self.feet]
        sole = position + quat_apply(quaternion, self.point.expand(env.num_envs, 2, 3))
        error = sole[..., :2] - env.scene.env_origins[:, None, :2] - self.ref_sole[frame, :, :2]
        # Fixed reference-phase entry, never reset when actual foot loses contact.
        drift, self.phase_offset = phase_drift(
            error, self.phase_offset, support, self.prev_expected, continuous)
        angular_error = (self.cmd.robot_body_ang_vel_w[:, self.feet]
                         - self.cmd.body_ang_vel_w[:, self.feet])
        smooth_valid = continuous[:, None] & self.prev_expected & support
        acceleration = torch.where(smooth_valid[..., None],
            (angular_error - self.prev_angular_error) / env.step_dt, 0.)
        bottom = sole_bottom_height(position, quaternion, self.offsets, self.radii)
        bottom -= env.scene.env_origins[:, 2, None]
        vz_error = torch.where(smooth_valid,
            (bottom - self.prev_bottom) / env.step_dt - self.target_vz[frame], 0.)
        costs = foot_costs(drift, angular_error, acceleration, bottom,
                          self.reference_bottom[frame], vz_error, support)
        weights = (drift_weight, rocking_weight, rocking_acc_weight,
                   clearance_weight, rising_weight)
        valid = (self.age > .15).float()
        ramp = support_ramp(env.common_step_counter, ramp_steps)
        single = torch.where(support.sum(-1) == 1, 1.2, 1.)
        raw = base + single * valid * sum(
            w * c.sum(-1) for w, c in zip(weights, costs.values()))
        self.geom_cache = ramp * bounded_support_cost(raw)
        self.cmd.metrics["t6_v2_raw_cost"] = raw
        self.cmd.metrics["t6_v2_bounded_cost"] = self.geom_cache
        self.cmd.metrics["t6_v2_ramp"] = torch.full_like(raw, ramp)
        for key, value in costs.items():
            self.cmd.metrics['t6_support_' + key] = value.sum(-1) * valid
        self.cmd.metrics['t6_support_drift_m'] = (drift.norm(dim=-1) * support).sum(-1) * valid
        self.prev_expected.copy_(support)
        self.prev_angular_error.copy_(angular_error)
        self.prev_bottom.copy_(bottom)
        self.geom_frame.copy_(frame)
        self.geom_step = env.common_step_counter
        return self.geom_cache

