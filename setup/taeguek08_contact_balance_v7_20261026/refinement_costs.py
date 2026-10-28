"""Additional costs only; established V3 tracking and contact costs stay active."""
import torch
from costs import robust

WEIGHTS = dict(total_peak=.45, total_rate=.04, total_excess=.30,
               foot_braking=.25, pelvis_braking=.30, early_height=.15,
               pelvis_height=.15, late_support=.30, support_gap=.25,
               quiet_drift=.08, quiet_speed=.08, quiet_rock=.04)

def window(t, lo, hi, edge=.10):
    def smooth(x):
        x=x.clamp(0,1)
        return x*x*(3-2*x)
    return smooth((t-lo)/edge)*smooth((hi-t)/edge)

def landing_gates(t):
    # First right-foot landing and the following step; later left-foot landing.
    return torch.stack((window(t,37.70,38.70),window(t,6.85,8.65)),dim=-1)

def total_impact(loads, body_weight, physics_dt):
    """Newest-first history; redistribution across feet cannot hide an impact."""
    if loads.ndim!=3 or loads.shape[1]!=2 or loads.shape[-1]<2:
        raise ValueError('Expected [N,2,K] newest-first load history, K>=2')
    total=loads.sum(1)/body_weight
    excess=(total[:,:-1]-2.5).clamp_min(0)
    rise=((total[:,:-1]-total[:,1:])/physics_dt-80.).clamp_min(0)/120.
    # No log compression of combined failures. RewardManager applies policy dt;
    # mean excess over substeps integrates to excess force / BW * physics dt.
    return robust(excess.amax(-1)),robust(rise).amax(-1),excess.mean(-1)

def approach_terms(bottom,down,ref_bottom,ref_vz,pelvis_z,ref_pelvis_z,
                   pelvis_vz,ref_pelvis_vz,gates,occupancy,expected):
    descending=(ref_vz<-.05).float()
    # Reference-only eligibility: lifting the actual foot cannot disable braking.
    near=((.20-ref_bottom)/.12).clamp(0,1)*descending*gates
    # Allow recorded airborne descent, then brake as the actual sole approaches
    # the plane. The 0.45 m/s touchdown allowance deliberately softens the fast
    # reference touchdown; airborne swing/arm tracking retains its V3 weights.
    envelope=torch.sqrt(.45**2+2*8.*bottom.clamp_min(0))
    allowed=torch.minimum((-ref_vz).clamp_min(0)+.15,envelope)
    braking=(robust((down-allowed).clamp_min(0)/.65)*near).sum(-1)
    early=(robust((ref_bottom-bottom-.008).clamp_min(0)/.025)*near).sum(-1)
    body_gate=near.amax(-1)
    # Pelvis body COM velocity, not whole-body COM. Do not demand a stationary
    # pelvis in a dynamic motion; penalize extra descent beyond the reference.
    body=robust((-pelvis_vz-(-ref_pelvis_vz).clamp_min(0)-.10).clamp_min(0)/.35)*body_gate
    height=robust((ref_pelvis_z-pelvis_z-.02).clamp_min(0)/.04)*body_gate
    late=(expected&(ref_bottom<.012)).float()*gates
    late_cost=((1-occupancy)+robust((bottom-ref_bottom-.005).clamp_min(0)/.015))*late
    return dict(foot_braking=braking,pelvis_braking=body,early_height=early,
                pelvis_height=height,late_support=late_cost.sum(-1))

def quiet_terms(t,drift,speed,ref_speed,omega_error,ref_tilt_rate,phase,expected):
    # Reference gates exclude intentional pivots, swings and fast relocations.
    quiet=((.06-ref_speed)/.04).clamp(0,1)*((.6-ref_tilt_rate)/.4).clamp(0,1)
    quiet=quiet*(phase==1)*expected
    emphasis=torch.ones_like(quiet)
    emphasis[:,1]+=3.*window(t,41.65,43.05)
    gate=quiet*emphasis
    return dict(
        quiet_drift=(robust((drift.norm(dim=-1)-.005).clamp_min(0)/.025)*gate).sum(-1),
        quiet_speed=(robust((speed-.015).clamp_min(0)/.10)*gate).sum(-1),
        quiet_rock=(robust((omega_error[...,:2].norm(dim=-1)-.25).clamp_min(0)/1.)*gate).sum(-1))

def support_gap(loads, body_weight, ref_speed, ref_tilt_rate, phase, expected,
                reference_age):
    """Substep low-load deficit in established quiet support, not a lift detector.

    Reference phase age grants 80 ms on entry. Actual contact loss cannot reset
    the gate. A small 0.04 BW floor allows weight transfer between the feet;
    pivots, flight and intentional fast reference movement remain unpenalized.
    """
    gate=((.06-ref_speed)/.04).clamp(0,1)*((.6-ref_tilt_rate)/.4).clamp(0,1)
    gate=gate*(phase==1)*expected*((reference_age-.04)/.04).clamp(0,1)
    deficit=((.04*body_weight-loads[:,:,:-1])/(.04*body_weight)).clamp(0,1)
    return (deficit.square().mean(-1)*gate).sum(-1)
