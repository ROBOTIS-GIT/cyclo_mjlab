"""The tracking task's observation, reward and termination (a port of blinkmimic/mdp.py and sensing.obs_noise).

The same analytic functions label simulator transitions and imagined world-model transitions. A state is the tuple
S = (p, R, v, w, q, qd): root position, root rotation matrix, world linear and angular velocity of the root link frame,
joint positions and joint velocities (BlinkMimic's joint order).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .motion import CONTROL_DT
from .rot import mat_to_6d, mat_to_quat, mm3, mv3, quat_to_mat, rot_angle_sq, tr3, yaw_mat

CONTACT_Z = 0.045
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT
BODY_POS_STD, BODY_ORI_STD, BODY_LINVEL_STD, BODY_ANGVEL_STD = 0.3, 0.4, 1.0, 3.14
ROOT_POS_STD, ROOT_ORI_STD = 0.3, 0.4
W_ACTION_RATE, W_JOINT_LIMIT, SOFT_LIMIT_FACTOR = -0.1, -2.0, 0.9
TERM_HEIGHT, TERM_TILT = 0.25, 0.8
# SuperMimic Table A.1 uniform observation-noise half-widths: q, qd, angular velocity, gravity, previous action, ref orientation
NOISE = (0.01, 0.5, 0.2, 0.05, 0.0, 0.05)


def state_dim(J: int) -> int:
    return 13 + 2 * J


def obs_dim(J: int) -> int:
    return 3 * J + 12


def inv_dim(J: int) -> int:
    return 13 + 2 * J


def flatten_state(S):
    p, R, v, w, q, qd = S
    return jnp.concatenate((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x):
    J = (x.shape[-1] - 13) // 2
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J :]


def invariant_features(S):
    """Heading- and position-invariant state: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    Rt = tr3(R)
    return jnp.concatenate((p[..., 2:3], mat_to_6d(mm3(tr3(yaw_mat(R)), R)), mv3(Rt, v), mv3(Rt, w), q, qd), -1)


class MDP:
    """Observation, reward (BeyondMimic's tracking terms plus InstantTrack's contact term), hard and soft termination, and
    the critic's privileged features."""

    def __init__(self, kin, default_q, feet: list[int], xi_dim: int):
        self.kin = kin
        J = len(default_q)
        self.J, self.xi_dim = J, xi_dim
        self.default_q = np.asarray(default_q, np.float32)
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * SOFT_LIMIT_FACTOR, mid + half * SOFT_LIMIT_FACTOR
        self.feet = list(feet)
        self.obs_dim = obs_dim(J)
        self.critic_extra = J + len(kin.tracked_idx) * 3 + 3 + 6 + xi_dim
        n = NOISE
        self.obs_noise = np.asarray(np.concatenate((np.full(J, n[0]), np.full(J, n[1]), np.full(3, n[2]), np.full(3, n[3]),
                                                     np.full(J, n[4]), np.full(6, n[5]))), np.float32)  # fmt: skip

    def obs(self, motion, S, a_prev, xi, f):
        """Noise-free observation: q - q0 - offset, qd, body angular velocity, projected gravity, a_prev, ref orientation (6D)."""
        p, R, v, w, q, qd = S
        Rt = tr3(R)
        dq0 = xi[..., 1 : 1 + self.J]
        rel = mat_to_6d(mm3(Rt, motion.root_R[f]))
        return jnp.concatenate((q - self.default_q - dq0, qd, mv3(Rt, w), -R[..., 2, :], a_prev, rel), -1)

    def add_noise(self, key, o):
        return o + (2 * jax.random.uniform(key, o.shape) - 1) * self.obs_noise

    def _anchored_ref(self, motion, p, R, f):
        rp, rR = motion.root_p[f], motion.root_R[f]
        dR = yaw_mat(mm3(R, tr3(rR)))
        anchor = jnp.concatenate((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor[..., None, :] + mv3(dR[..., None, :, :], motion.body_p[f] - rp[..., None, :])
        return rp, rR, dR, ref_bp

    def reward(self, motion, S, f, a, a_prev, w_contact: float = 0.0, terms: bool = False):
        """Per-step reward (times dt) of S against reference frame f; w_contact > 0 adds InstantTrack's contact term."""
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR, dR, ref_bp = self._anchored_ref(motion, p, R, f)
        ref_bR = mm3(dR[..., None, :, :], motion.body_R[f])
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((motion.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((motion.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (jnp.exp(-e_pos / BODY_POS_STD**2) + jnp.exp(-e_ori / BODY_ORI_STD**2) + jnp.exp(-e_lv / BODY_LINVEL_STD**2)
                   + jnp.exp(-e_av / BODY_ANGVEL_STD**2) + 0.5 * jnp.exp(-e_rp / ROOT_POS_STD**2) + 0.5 * jnp.exp(-e_ro / ROOT_ORI_STD**2))  # fmt: skip
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (jax.nn.relu(q - self.soft_hi) + jax.nn.relu(self.soft_lo - q)).sum(-1)
        r = (r_track + W_ACTION_RATE * r_rate + W_JOINT_LIMIT * r_lim) * CONTROL_DT
        if w_contact > 0:
            fz = bp[..., self.feet, 2]
            c_sim = jax.nn.sigmoid((CONTACT_Z - fz) / 0.005)
            c_ref = motion.ref_contact[f]
            slip = (bv[..., self.feet, :2] ** 2).sum(-1)
            r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * jnp.exp(-slip / 0.3**2)
            r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
        if terms:
            return r, {"e_body_pos": e_pos, "e_body_ori": e_ori, "e_root_pos": e_rp, "track": r_track * CONTROL_DT}
        return r

    def termination(self, motion, S, f):
        """Root height or tilt too far from the reference."""
        p, R = S[0], S[1]
        e_z = jnp.abs(motion.root_p[f][..., 2] - p[..., 2])
        e_tilt = jnp.abs(motion.root_R[f][..., 2, 2] - R[..., 2, 2])
        return (e_z > TERM_HEIGHT) | (e_tilt > TERM_TILT)

    def soft_termination(self, motion, S, f, tz: float = 0.02, tt: float = 0.04):
        """A differentiable termination probability for imagined rollouts (sigmoids around the hard thresholds)."""
        p, R = S[0], S[1]
        e_z = jnp.abs(motion.root_p[f][..., 2] - p[..., 2])
        e_tilt = jnp.abs(motion.root_R[f][..., 2, 2] - R[..., 2, 2])
        return 1 - (1 - jax.nn.sigmoid((e_z - TERM_HEIGHT) / tz)) * (1 - jax.nn.sigmoid((e_tilt - TERM_TILT) / tt))

    def critic_features(self, motion, S, f, xi_n):
        """Invariant state, q_ref - q, body and root errors in the heading frame, ref orientation (6D), sim context."""
        p, R, v, w, q, qd = S
        Ht = tr3(yaw_mat(R))
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        rp, rR, _, ref_bp = self._anchored_ref(motion, p, R, f)
        e_body = mv3(Ht[..., None, :, :], ref_bp - bp).reshape(bp.shape[:-2] + (-1,))
        e_root = mv3(Ht, rp - p)
        rel = mat_to_6d(mm3(tr3(R), rR))
        return jnp.concatenate((invariant_features(S), motion.q[f] - q, e_body, e_root, rel, xi_n), -1)
