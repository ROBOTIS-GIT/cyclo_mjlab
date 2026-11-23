# Copyright 2025 The mjlab Developers
# SPDX-License-Identifier: Apache-2.0
# Source: HybridRobotics/mjlab_research dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
# src/mjlab/utils/forward_kinematics.py; implementation unchanged.
"""Forward kinematics: world-frame pose and velocity of a floating-base robot's
bodies from its root and joint state, differentiable, in the convention of mjlab's
``body_link_pose_w`` and ``body_link_vel_w``.

Each body follows MuJoCo's kinematics: the parent frame, then the body's fixed
offset, then each of its hinges, turning ``qpos - qpos0`` about the hinge axis
through the hinge anchor. Every hinge, and every body without one, is a step.
Each returned body composes the steps on its path from the root, root first, and
sums their hinge twists, taken at the root origin; the root pose and twist apply
last. The pass runs on per-component ``(..., R)`` tensors, which a compiler fuses
into few kernels.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

Vec = tuple[torch.Tensor, ...]


class ForwardKinematics:
  """The steps on the paths from the entity's free-joint root to the bodies at
  ``body_rows``, as constants, and the batched pass along them.

  ``body_ids`` are the robot's global body ids in entity order, the first one the
  free-joint root, ``joint_q_adr`` the qpos addresses in joint column order,
  ``body_rows`` the entity slots returned, in output order. ``joint_names`` are the
  names of the joints at ``joint_q_adr`` in column order, without scene prefixes.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    body_ids: torch.Tensor,
    joint_q_adr: torch.Tensor,
    body_rows: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
  ):
    rows = [int(body_ids[row]) for row in body_rows]
    col_of = {int(adr): col for col, adr in enumerate(joint_q_adr)}
    tree = set()
    for body in rows:
      while body and body not in tree:
        tree.add(body)
        body = int(mj_model.body_parentid[body])
    order = sorted(tree)  # MuJoCo numbers parents before children

    free, hinge = mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE
    zero, identity = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    # (parent, joint column, qpos0, axis, anchor, offset pos, offset quat); a step
    # without a hinge reads column J, a zero pad after the J joints. Step 0, the
    # root, is the identity.
    fixed = (len(col_of), 0.0, zero, zero)
    steps: list[tuple] = [(0, *fixed, zero, identity)]
    step_of = {order[0]: 0}
    for body in order:
      parent, first = int(mj_model.body_parentid[body]), int(mj_model.body_jntadr[body])
      joints = range(first, first + int(mj_model.body_jntnum[body]))
      kinds = [mj_model.jnt_type[j] for j in joints]
      root = body == int(body_ids[0])
      if (parent == 0) != root or kinds != ([free] if root else [hinge] * len(kinds)):
        raise ValueError(
          f"body {mj_model.body(body).name}: needs the entity's first body as its "
          "free-joint root under the world, with only hinges below it"
        )
      if root:
        continue
      adrs = [int(mj_model.jnt_qposadr[j]) for j in joints]
      hinges = [
        (col_of[a], mj_model.qpos0[a], mj_model.jnt_axis[j], mj_model.jnt_pos[j])
        for j, a in zip(joints, adrs, strict=True)
      ]
      step = step_of[parent]
      offset = mj_model.body_pos[body], mj_model.body_quat[body]
      for joint in hinges or [fixed]:
        steps.append((step, *joint, *offset))
        step, offset = len(steps) - 1, (zero, identity)
      step_of[body] = step

    # Each returned body's path, root first, front-padded with the root step to
    # one length.
    paths = []
    for body in rows:
      step, path = step_of[body], []
      while step:
        path.append(step)
        step = steps[step][0]
      paths.append(path[::-1])
    depth = max(map(len, paths))
    paths = [[0] * (depth - len(path)) + path for path in paths]
    # Per path step, with o its offset quaternion: the hinge anchor and axis in the
    # parent frame, the anchor in the step frame, o, and o * (0, axis); the step's
    # quaternion is cos(h) o + sin(h) o * (0, axis) for h = (qpos - qpos0) / 2.
    consts = np.zeros((depth, len(rows), 17))
    for r, path in enumerate(paths):
      for d, step in enumerate(path):
        _, _, _, axis, anchor, offset_pos, offset_quat = steps[step]
        parent_anchor, parent_axis, turned = np.zeros(3), np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(parent_anchor, anchor, offset_quat)
        mujoco.mju_rotVecQuat(parent_axis, axis, offset_quat)
        mujoco.mju_mulQuat(turned, offset_quat, np.concatenate([[0.0], axis]))
        consts[d, r] = np.concatenate(
          [offset_pos + parent_anchor, parent_axis, anchor, offset_quat, turned]
        )
    self.consts = torch.as_tensor(consts, device=device, dtype=dtype)
    by_step = [[steps[s] for s in path] for path in zip(*paths, strict=True)]
    self.col = torch.tensor([[s[1] for s in d] for d in by_step], device=device)
    self.qpos0 = torch.tensor(
      [[s[2] for s in d] for d in by_step], device=device, dtype=dtype
    )
    names = {
      int(adr): mj_model.joint(j).name.split("/")[-1]
      for j, adr in enumerate(mj_model.jnt_qposadr)
    }
    self.joint_names = [names[adr] for adr in col_of if adr in names]

  def __call__(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Positions ``(..., R, 3)``, wxyz quaternions ``(..., R, 4)``, linear and
    angular velocities ``(..., R, 3)`` of the ``R`` bodies, in the world frame,
    from the root link's world pose and velocity and the joints ``(..., J)``."""
    joint_pos, joint_vel = (
      torch.nn.functional.pad(x, (0, 1)) for x in (joint_pos, joint_vel)
    )
    zero, one = root_pos.new_zeros(()), root_pos.new_ones(())
    pos: Vec = (zero, zero, zero)
    quat: Vec = (one, zero, zero, zero)
    lin: Vec = (zero, zero, zero)
    ang: Vec = (zero, zero, zero)
    for col, qpos0, consts in zip(self.col, self.qpos0, self.consts, strict=True):
      c = consts.unbind(-1)
      half = (joint_pos[..., col] - qpos0) / 2
      # The hinge anchor and twist in the root frame, from the parent's pose.
      anchor = _add(pos, _rotate(quat, c[:3]))
      twist = tuple(a * joint_vel[..., col] for a in _rotate(quat, c[3:6]))
      turn = tuple(
        half.cos() * o + half.sin() * t for o, t in zip(c[9:13], c[13:], strict=True)
      )
      quat = _mul(quat, turn)
      pos = _sub(anchor, _rotate(quat, c[6:9]))
      ang = _add(ang, twist)
      lin = _add(lin, _cross(anchor, twist))

    w, x, y, z = root_quat.unsqueeze(-2).unbind(-1)
    norm = (w * w + x * x + y * y + z * z).sqrt()
    root = (w / norm, x / norm, y / norm, z / norm)
    pos = _rotate(root, pos)
    ang = _add(root_ang_vel.unsqueeze(-2).unbind(-1), _rotate(root, ang))
    lin = _add(root_lin_vel.unsqueeze(-2).unbind(-1), _rotate(root, lin))
    lin = _add(lin, _cross(ang, pos))
    pos = _add(root_pos.unsqueeze(-2).unbind(-1), pos)
    state = torch.stack([*pos, *_mul(root, quat), *lin, *ang], dim=-1)
    pos_w, quat_w, lin_vel_w, ang_vel_w = state.split((3, 4, 3, 3), dim=-1)
    return pos_w, quat_w, lin_vel_w, ang_vel_w


def _add(a: Vec, b: Vec) -> Vec:
  return tuple(x + y for x, y in zip(a, b, strict=True))


def _sub(a: Vec, b: Vec) -> Vec:
  return tuple(x - y for x, y in zip(a, b, strict=True))


def _cross(a: Vec, b: Vec) -> Vec:
  return (
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  )


def _rotate(q: Vec, v: Vec) -> Vec:
  """``v`` turned by the unit wxyz quaternion ``q``."""
  t = tuple(2 * x for x in _cross(q[1:], v))
  return _add(_add(v, tuple(q[0] * x for x in t)), _cross(q[1:], t))


def _mul(a: Vec, b: Vec) -> Vec:
  """The Hamilton product of wxyz quaternions."""
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return (
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  )
