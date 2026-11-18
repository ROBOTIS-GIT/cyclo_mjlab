"""Batched rotations in JAX (quaternions wxyz, matrices act on column vectors), ported from blinkmimic/rotations.py.

3x3 products are written elementwise (multiply + sum), so XLA fuses them into the surrounding kernels instead of emitting
tiny batched matmuls (InstantTrack's --fused_rot, here the only form).
"""

import jax.numpy as jnp


def mm3(A, B):
    return (A[..., :, :, None] * B[..., None, :, :]).sum(-2)


def mv3(A, v):
    return (A * v[..., None, :]).sum(-1)


def tr3(A):
    return jnp.swapaxes(A, -1, -2)


def normalize(v, eps=1e-12):
    return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), eps)


def quat_to_mat(q):
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(m):
    """Robust matrix -> quaternion (wxyz, w >= 0): the largest-diagonal branch, as blinkmimic.rotations.mat_to_quat."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = jnp.sqrt(jnp.maximum(1 + m00 + m11 + m22, 1e-10)) / 2
    qx = jnp.sqrt(jnp.maximum(1 + m00 - m11 - m22, 1e-10)) / 2
    qy = jnp.sqrt(jnp.maximum(1 - m00 + m11 - m22, 1e-10)) / 2
    qz = jnp.sqrt(jnp.maximum(1 - m00 - m11 + m22, 1e-10)) / 2
    cands = jnp.stack(
        (
            jnp.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            jnp.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            jnp.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            jnp.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        axis=-2,
    )
    idx = jnp.argmax(jnp.stack((qw, qx, qy, qz), -1), -1)
    q = jnp.take_along_axis(cands, idx[..., None, None], axis=-2)[..., 0, :]
    q = normalize(q)
    return q * jnp.where(q[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis, angle):
    """Rodrigues for a fixed unit axis (3,) and angles (...)."""
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = jnp.cos(angle), jnp.sin(angle)
    C = 1 - c
    return jnp.stack(
        (
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ),
        axis=-1,
    ).reshape(angle.shape + (3, 3))  # fmt: skip


def skew(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = jnp.zeros_like(x)
    return jnp.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v):
    """Rotation vector (..., 3) -> matrix, smooth at zero."""
    theta2 = (v * v).sum(-1, keepdims=True)
    theta = jnp.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = jnp.where(small, 1 - theta2 / 6, jnp.sin(theta) / theta)
    b = jnp.where(small, 0.5 - theta2 / 24, (1 - jnp.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    return jnp.eye(3, dtype=v.dtype) + a[..., None] * K + b[..., None] * mm3(K, K)


def quat_to_rotvec(q):
    q = q * jnp.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = jnp.linalg.norm(v, axis=-1, keepdims=True)
    half = jnp.arctan2(sin_half, q[..., :1])
    scale = jnp.where(sin_half < 1e-6, 2.0 / jnp.maximum(q[..., :1], 1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def so3_log(R):
    return quat_to_rotvec(mat_to_quat(R))


def rot_angle_sq(Ra, Rb):
    """Squared geodesic angle between rotations (acos of the trace, clamped)."""
    tr = (Ra * Rb).sum((-1, -2))
    cos = jnp.clip((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return jnp.arccos(cos) ** 2


def yaw_mat(R):
    """Heading (yaw-only) rotation of R, from the projected x axis."""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = jnp.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


def mat_to_6d(R):
    """First two columns, flattened row-major (as blinkmimic: R[..., :, :2].reshape(..., 6))."""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))
