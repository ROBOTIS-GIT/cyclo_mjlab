"""Pure tensor objectives, separated for behavioral tests."""
import torch


def robust(x):
    return torch.sqrt(1 + x.square()) - 1


def compress(raw, scale=2.):
    # Sublinear tail without a hard ceiling: large failures remain distinguishable.
    return scale * torch.log1p(raw / scale)


def phase_drift(error, offset, expected, previous, continuous):
    start = expected & (~previous | ~continuous[:, None])
    offset = torch.where(start[..., None], error, offset)
    return error - offset, offset


def geometric_costs(drift, speed, ref_speed, bottom, ref_bottom, omega_error, phase, expected):
    planted = expected & (phase == 1)
    pivot = expected & (phase == 2)
    # A pivot may move the sole centre. Position locking applies only to planted feet.
    drift_cost = robust((drift.norm(dim=-1)-.015).clamp_min(0)/.04) * planted
    allowed = torch.where(pivot, ref_speed.clamp(max=.25)+.05, torch.full_like(speed, .04))
    motion_cost = robust((speed-allowed).clamp_min(0)/.20) * expected
    height_cost = robust((bottom-ref_bottom.clamp_min(0)-.003).clamp_min(0)/.01) * expected
    rock_cost = robust((omega_error[..., :2].norm(dim=-1)-.5).clamp_min(0)/2) * expected
    return drift_cost, motion_cost, height_cost, rock_cost


def impact_costs(loads, weight_n, physics_dt):
    """Newest-first substep history; only new samples plus one previous sample."""
    now = loads[:, :, :-1]
    normalized = now / weight_n
    peak = robust((normalized.amax(-1)-2.5).clamp_min(0)).sum(-1)
    peak += robust((normalized.sum(1).amax(-1)-3.5).clamp_min(0)/1.5)
    rise = ((loads[:, :, :-1]-loads[:, :, 1:])/(weight_n*physics_dt)).clamp_min(0)
    rate = robust((rise-80).clamp_min(0)/120).amax(-1).sum(-1)
    # Only force ABOVE a loading allowance, not the impulse needed to support mass.
    excess = (normalized.sum(1)-3.5).clamp_min(0).mean(-1)
    return peak, rate, excess


def approach_cost(speed_down, bottom, ref_bottom):
    gate = (bottom < .06) & (bottom > -.02) & (ref_bottom < .10)
    return (robust((speed_down-.6).clamp_min(0)/.6)*gate).sum(-1)


def hold_costs(drift, speed, gravity, ref_gravity, ref_speed, ref_tilt_rate,
               phase, expected, ref_bottom):
    """Quiet planted translation; yaw-invariant tilt tracking also during pivots.

    Eligibility depends only on reference, never actual contact or actual speed.
    Local gravity measures roll/pitch without suppressing the intended yaw turn.
    """
    quiet = ((.06-ref_speed)/.04).clamp(0,1)*((.6-ref_tilt_rate)/.4).clamp(0,1)
    quiet = quiet*(phase==1)*expected
    displacement = robust((drift.norm(dim=-1)-.005).clamp_min(0)/.025)*quiet
    motion = robust((speed-.015).clamp_min(0)/.10)*quiet
    # Include low supporting feet during a kick even if short phase fragments
    # were omitted by the conservative contact label. Track, do not freeze, tilt.
    tilt_support = expected | ((ref_bottom<.012)&(ref_bottom.flip(-1)>.25))
    chord = (gravity-ref_gravity).norm(dim=-1)
    tilt = robust((chord-.035).clamp_min(0)/.087)*tilt_support
    return displacement, motion, tilt, quiet


def swing_costs(actual_xy, reference_xy, bottom, ref_bottom, phase):
    """Preserve swing trajectory and absolute sole height, away from touchdown."""
    gate = ((ref_bottom-.08)/.17).clamp(0,1)*(phase==0)
    xy = robust((actual_xy-reference_xy).norm(dim=-1)/.08)*gate
    height = robust((bottom-ref_bottom)/.06)*gate
    return xy, height, gate
