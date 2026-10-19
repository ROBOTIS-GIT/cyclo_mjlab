# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Conservative reference support labels for Taegeuk's low-clearance steps."""
import math
import torch
from mjlab.utils.lab_api.math import quat_apply
from .support_contacts import FootSupportQuality


def support_phase_mask(grounded, bottom, lowpoint_speed, fps,
                       slow_speed=0.20, opposite_clearance=0.15,
                       min_duration_s=0.10, boundary_margin_s=0.04):
  """Keep quiet support and kick pivots; exclude low, fast foot relocation.

  Heights are relative to each reference foot's minimum. These are confidence
  labels, not physical contact ground truth. Opposite high-foot support retains
  pivots even when the supporting reference slides. No actual robot state is
  used to choose eligibility, so the policy cannot escape by moving faster.
  """
  relative = bottom - bottom.amin(dim=0)
  opposite_high = relative.flip(-1) > opposite_clearance
  candidate = grounded & ((lowpoint_speed < slow_speed) | opposite_high)
  # Offline preprocessing once per environment construction, not per RL step.
  raw = candidate.cpu(); result = torch.zeros_like(raw)
  minimum = max(1, math.ceil(min_duration_s * fps - 1e-8))
  margin = max(0, math.ceil(boundary_margin_s * fps - 1e-8))
  for foot in range(2):
    padded = torch.cat((torch.zeros(1,dtype=torch.bool),raw[:,foot],torch.zeros(1,dtype=torch.bool)))
    change = torch.diff(padded.to(torch.int8))
    starts = (change == 1).nonzero().flatten().tolist()
    ends = (change == -1).nonzero().flatten().tolist()
    for a,b in zip(starts,ends):
      # Preserve the true start/end of the clip; trim only internal transitions.
      lo = a + (margin if a > 0 else 0)
      hi = b - (margin if b < len(raw) else 0)
      if b-a >= minimum and hi-lo >= minimum:
        result[lo:hi,foot] = True
  return result.to(grounded.device)


class TaegeukFootSupportQuality(FootSupportQuality):
  """Reuse the tested contact tracker/cost, replacing only phase eligibility."""
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    ref=self.cmd.reference
    p=ref.body_position_w[:,self.feet]
    q=ref.body_orientation_w[:,self.feet]
    n,feet,_=q.shape;count=self.offsets.shape[1]
    rotated=quat_apply(q[:,:,None,:].expand(n,feet,count,4),
                      self.offsets[None].expand(n,-1,-1,-1))
    surface=rotated.clone();surface[...,2]-=self.radii[None]
    bottom=p[...,2]+surface[...,2].amin(-1)
    com=quat_apply(q,self.com[None].expand(n,-1,-1))
    v=ref.body_linear_velocity_w[:,self.feet,None,:]+torch.cross(
      ref.body_angular_velocity_w[:,self.feet,None,:].expand_as(surface),
      surface-com[:,:,None,:],dim=-1)
    low=surface[...,2] <= surface[...,2].amin(-1,keepdim=True)+0.005
    # The slowest low material point permits a planted pivot; a translating
    # flat foot has no stationary low point and is excluded when both feet low.
    min_speed=torch.where(low,v[...,:2].norm(dim=-1),torch.inf).amin(-1)
    self.original_reference_support=self.reference_support
    self.reference_support=support_phase_mask(self.reference_support,bottom,min_speed,float(ref.fps))
    self.reference_bottom=bottom
    self.reference_lowpoint_speed=min_speed
