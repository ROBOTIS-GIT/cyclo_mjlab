"""Reference-gated pivot support, bounded touchdown speed and path recovery."""
import torch
from costs import robust
from refinement_costs import window

WEIGHTS = dict(support_slip=.08, support_gap=.12, support_height=.12,
               support_tilt=.06, path_recovery=.12,
               first_foot_braking=.15, first_pelvis_braking=.10)


def support_gate(bottom, lowpoint_speed, phase, expected, reference_age):
    # No yaw restriction: a stationary material point can support a pivot.
    eligible = expected & ((phase == 1) | (phase == 2)) & (bottom < .015)
    confidence = ((.20-lowpoint_speed)/.10).clamp(0, 1)
    grace = (reference_age/.08).clamp(0, 1)
    return eligible*confidence*grace


def support_terms(gate, speed, ref_speed, loads, bw, bottom, ref_bottom,
                  gravity, ref_gravity):
    allowance = .005+ref_speed.clamp(0, .010)
    slip = robust((speed-allowance).clamp_min(0)/.020)
    # A small floor permits dual-support load transfer. Reference eligibility
    # and age cannot be disabled by an actual contact loss.
    gap = ((.02*bw-loads[:, :, :-1])/(.02*bw)).clamp(0, 1).square().mean(-1)
    height = robust((bottom-ref_bottom.clamp_min(0)-.003).clamp_min(0)/.010)
    tilt = robust(((gravity-ref_gravity).norm(dim=-1)-.03).clamp_min(0)/.08)
    return dict(support_slip=(slip*gate).sum(-1),
                support_gap=(gap*gate).sum(-1),
                support_height=(height*gate).sum(-1),
                support_tilt=(tilt*gate).sum(-1))


def recovery_terms(t, actual_xy, ref_xy, bottom, down, ref_bottom, ref_vz,
                   pelvis_vz, ref_pelvis_vz):
    # Remains informative when far from the reference, unlike an exponential
    # position score. Do not reward a different body path to reduce contact cost.
    path = robust(((actual_xy-ref_xy).norm(dim=-1)-.03).clamp_min(0)/.15)
    near = window(t, 6.90, 7.55)*((.20-ref_bottom[:, 1])/.12).clamp(0, 1)*(ref_vz[:, 1] < -.05)
    envelope = torch.sqrt(.35**2+2*6.*bottom[:, 1].clamp_min(0))
    allowance = torch.minimum((-ref_vz[:, 1]).clamp_min(0)+.15, envelope)
    foot = robust((down[:, 1]-allowance).clamp_min(0)/.65)*near
    pelvis = robust((-pelvis_vz-(-ref_pelvis_vz).clamp_min(0)-.05).clamp_min(0)/.35)*near
    return dict(path_recovery=path, first_foot_braking=foot,
                first_pelvis_braking=pelvis)
