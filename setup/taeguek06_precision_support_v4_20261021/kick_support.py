"""Small kick-support corrections outside the mature BalancedV1 cost cap.

No new contact state machine. The existing support term remains registered once.
Geometry/reference alone choose eligibility; physical flight cannot disable it.
"""
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from source.tasks.mimic.mdp.taeguek_support import support_phase_mask
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height
from support import SupportStability, huber


def correction_ramp(step, ramp_steps=72000):
    return max(0., min(1., step / max(1, ramp_steps)))


def kick_support_mask(bottom, fps):
    """Opposite foot clearly airborne, support foot near the reference floor.

    Do not use noisy reference vertical velocity: millimetre jitter must not
    remove a support objective. Retain genuine pivots, trim transitions 40 ms.
    """
    relative = bottom - bottom.amin(0)
    candidate = (relative < .018) & (relative.flip(-1) > .25)
    return support_phase_mask(candidate, bottom, torch.zeros_like(bottom), fps,
                              min_duration_s=.12, boundary_margin_s=.04)


def correction_costs(bottom, gravity, target_gravity, angular, target_angular,
                     loads, weight_n, expected):
    # Absolute ground clearance, not reference + 10 mm. All nine sphere bottoms
    # are considered by sole_bottom_height; rolling on a toe is not a hop.
    height = huber((bottom - .001).clamp_min(0) / .008) * expected
    tilt = huber(((gravity - target_gravity).norm(dim=-1) - .035).clamp_min(0) / .087) * expected
    # Permit intended reference tilt rates and any world-z pivot rotation.
    rocking = huber((angular[..., :2].norm(dim=-1)
                    - target_angular[..., :2].norm(dim=-1) - .20).clamp_min(0)) * expected
    # A soft floor of 0.2 body weight, not full weight on every support sample.
    # Only the known kick-support foot is selected; no double-stance load split.
    deficit = ((.20 - loads / weight_n).clamp_min(0) / .20).square().mean(-1) * expected
    return dict(height=height, tilt=tilt, rocking=rocking, deficit=deficit)


class KickSupportStability(SupportStability):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ref = self.cmd.reference
        self.kick_support = kick_support_mask(self.reference_bottom, float(ref.fps))
        self.up = torch.tensor([0., 0., 1.], device=env.device)
        self.target_gravity = quat_apply_inverse(
            ref.body_orientation_w[:, self.feet], self.up.expand(ref.num_frames, 2, 3))
        self.weight_n = float(env.sim.mj_model.body_mass.sum()) * 9.81
        self.correction_step = -1
        self.correction_cache = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, 'correction_cache'):
            ids = slice(None) if env_ids is None else env_ids
            self.correction_cache[ids] = 0
            self.correction_step = -1

    def __call__(self, env, new_ramp_steps=72000):
        if self.correction_step == env.common_step_counter:
            return self.correction_cache
        # Already-trained support terms stay fully active from the first update.
        base = super().__call__(env, ramp_steps=1)
        c = self.cmd; f = c.frame_ids; expected = self.kick_support[f]
        q = c.robot_body_quat_w[:, self.feet]
        bottom = sole_bottom_height(c.robot_body_pos_w[:, self.feet], q, self.offsets, self.radii)
        bottom = bottom - env.scene.env_origins[:, 2, None]
        gravity = quat_apply_inverse(q, self.up.expand(env.num_envs, 2, 3))
        costs = correction_costs(bottom, gravity, self.target_gravity[f],
            c.robot_body_ang_vel_w[:, self.feet], c.body_ang_vel_w[:, self.feet],
            foot_load_history(env)[:, :, :env.cfg.decimation], self.weight_n, expected)
        valid = (self.age > .15).float()
        extra = sum(weight * costs[name].sum(-1) for name, weight in
                    [('height', .12), ('tilt', .12), ('rocking', .06), ('deficit', .30)]) * valid
        ramp = correction_ramp(env.common_step_counter, new_ramp_steps)
        # No outer tanh: the existing 0.6 cap must not hide the small correction.
        self.correction_cache = base + ramp * extra
        for key, value in costs.items():
            c.metrics['t6_kick_support_' + key] = value.sum(-1) * valid
        c.metrics['t6_kick_support_ramp'] = torch.full_like(extra, ramp)
        c.metrics['t6_kick_support_extra'] = ramp * extra
        c.metrics['t6_kick_support_expected'] = expected.float().sum(-1)
        self.correction_step = env.common_step_counter
        return self.correction_cache
