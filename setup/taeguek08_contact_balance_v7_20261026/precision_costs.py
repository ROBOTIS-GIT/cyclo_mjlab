"""Complement mature V5 rewards without re-ramping the learned objectives."""
import math
import torch
from costs import robust
from refinement_costs import total_impact,landing_gates

WEIGHTS=dict(outside_peak=.45,outside_rate=.04,outside_excess=.30,
             locked_drift=.08,locked_speed=.08,locked_load_change=.04)

def stable_mask(bottom,sole_speed,yaw_rate,tilt_rate,phase,expected,fps):
    candidate=((bottom<.012)&(sole_speed<.05)&(yaw_rate.abs()<.5)&
               (tilt_rate<.6)&(phase==1)&expected).cpu()
    out=torch.zeros_like(candidate)
    margin=math.ceil(.06*fps);minimum=math.ceil(.30*fps)
    for foot in range(2):
        change=torch.diff(torch.cat((torch.tensor([False]),candidate[:,foot],torch.tensor([False]))).int())
        for a,b in zip((change==1).nonzero().flatten().tolist(),(change==-1).nonzero().flatten().tolist()):
            if b-a>=minimum:out[a+margin:b-margin,foot]=True
    return out.to(bottom.device)

def costs(t,loads,bw,physics_dt,locked,drift,speed):
    # Existing refinement already applies these exact weights inside the two
    # landing windows. Their complement makes total-force coverage uniform.
    outside=1-landing_gates(t).amax(-1)
    peak,rate,excess=total_impact(loads,bw,physics_dt)
    jumps=((loads[:,:,:-1]-loads[:,:,1:]).abs()/bw-.10).clamp_min(0)
    return dict(outside_peak=peak*outside,outside_rate=rate*outside,
                outside_excess=excess*outside,
                locked_drift=(robust((drift.norm(dim=-1)-.003).clamp_min(0)/.02)*locked).sum(-1),
                locked_speed=(robust((speed-.005).clamp_min(0)/.05)*locked).sum(-1),
                locked_load_change=(robust(jumps/.20).mean(-1)*locked).sum(-1))
