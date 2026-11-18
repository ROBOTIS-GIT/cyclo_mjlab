"""Differentiable forward kinematics of the G1, ported from blinkmimic/kinematics.py (tree and offsets from the export's meta)."""

import jax.numpy as jnp
import numpy as np

from .rot import axis_angle_to_mat, mm3, mv3


class Kinematics:
    def __init__(self, meta: dict):
        self.parent = list(meta["kin_parent"])
        self.joint_idx = list(meta["kin_joint_idx"])
        self.axes = np.asarray(meta["kin_axes"], np.float32)
        self.off_p = np.asarray(meta["kin_off_p"], np.float32)
        self.off_R = np.asarray(meta["kin_off_R"], np.float32)
        self.tracked_idx = list(meta["kin_tracked_idx"])
        self.num_bodies = len(self.parent)
        self.q_lo = np.asarray(meta["q_lo"], np.float32)
        self.q_hi = np.asarray(meta["q_hi"], np.float32)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions (..., n, 3), rotations (..., n, 3, 3), link-origin linear and angular velocities."""
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + mv3(Rp, self.off_p[i])
            R_off = mm3(Rp, self.off_R[i])
            Ri = mm3(R_off, axis_angle_to_mat(self.axes[i], q[..., j]))
            wi = W[par] + mv3(R_off, self.axes[i]) * qd[..., j : j + 1]
            vi = V[par] + jnp.cross(W[par], pi - P[par])
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (jnp.stack([P[i] for i in idx], -2), jnp.stack([Rs[i] for i in idx], -3),
                jnp.stack([V[i] for i in idx], -2), jnp.stack([W[i] for i in idx], -2))  # fmt: skip
