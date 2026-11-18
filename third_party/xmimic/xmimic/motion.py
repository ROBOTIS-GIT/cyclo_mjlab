"""Reference motion library (a port of blinkmimic/motion.py).

The G1-retargeted LAFAN1 clips (30 fps CSV) are resampled to the 50 Hz control rate. Root and joint velocities come from
finite differences, the tracked bodies from forward kinematics. Clips are cut into 1-second reset bins, and the policy's
goal is a window of future reference frames. The library is a pytree of device arrays: it is passed into jitted functions
as an argument and never closed over as a constant.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
from pathlib import Path
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .rot import mat_to_6d, mm3, mv3, tr3, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50
GOAL_Q_OFFSET = 15  # index of q(f+1) in the goal vector (first future frame: dp 3, rot 6, v 3, w 3, then q)


def load_lafan1_csv(path: str):
    """30 fps rows of root position, root quaternion (xyzw) and 29 joint angles -> (p, quat wxyz, q, fps)."""
    raw = np.loadtxt(path, delimiter=",")
    p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
    return p, np.concatenate((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1), j, 30.0


def _slerp_np(q0, q1, t):
    """blinkmimic.rotations.quat_slerp in float32 numpy."""
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.minimum(np.abs(dot), np.float32(1.0))
    theta = np.arccos(dot)
    s = np.sin(theta)
    t = t[..., None]
    w0 = np.where(s < 1e-6, 1 - t, np.sin((1 - t) * theta) / (s + np.float32(1e-12)))
    w1 = np.where(s < 1e-6, t, np.sin(t * theta) / (s + np.float32(1e-12)))
    q = w0 * q0 + w1 * q1
    return (q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), np.float32(1e-12))).astype(np.float32)


def _quat_to_mat_np(q):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def _so3_log_np(R):
    """blinkmimic.rotations.so3_log (largest-diagonal quaternion branch, then the rotation vector) in numpy."""
    m = R
    qw = np.sqrt(np.maximum(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], 1e-10)) / 2
    qx = np.sqrt(np.maximum(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-10)) / 2
    qy = np.sqrt(np.maximum(1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 1e-10)) / 2
    qz = np.sqrt(np.maximum(1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], 1e-10)) / 2
    m01, m02, m10, m12, m20, m21 = m[..., 0, 1], m[..., 0, 2], m[..., 1, 0], m[..., 1, 2], m[..., 2, 0], m[..., 2, 1]
    cands = np.stack((np.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
                      np.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
                      np.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
                      np.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1)), -2)  # fmt: skip
    idx = np.argmax(np.stack((qw, qx, qy, qz), -1), -1)
    q = np.take_along_axis(cands, idx[..., None, None], axis=-2)[..., 0, :]
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = np.linalg.norm(v, axis=-1, keepdims=True)
    half = np.arctan2(sin_half, q[..., :1])
    scale = np.where(sin_half < 1e-6, 2.0 / np.maximum(q[..., :1], 1e-6), 2 * half / (sin_half + 1e-12))
    return (v * scale).astype(np.float32)


def _gradient(x, dt):
    """Central differences inside, one-sided at the ends (numpy.gradient in float32)."""
    g = np.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / np.float32(2 * dt)
    g[0] = (x[1] - x[0]) / np.float32(dt)
    g[-1] = (x[-1] - x[-2]) / np.float32(dt)
    return g


@partial(jax.tree_util.register_dataclass,
         data_fields=["root_p", "root_R", "root_v", "root_w", "q", "qd", "body_p", "body_R", "body_v", "body_w", "root_v_local",
                      "root_w_local", "clip_id", "clip_start", "clip_end", "bin_clip", "bin_start", "frame_bin", "ref_contact"],
         meta_fields=["names", "num_frames", "num_bins", "num_joints"])  # fmt: skip
@dataclasses.dataclass
class Motion:
    """All clips concatenated along one frame axis. Per frame: root and joint state, tracked-body poses and velocities, the
    clip it belongs to and its reset bin, and the reference foot contacts."""

    root_p: jax.Array
    root_R: jax.Array
    root_v: jax.Array
    root_w: jax.Array
    q: jax.Array
    qd: jax.Array
    body_p: jax.Array
    body_R: jax.Array
    body_v: jax.Array
    body_w: jax.Array
    root_v_local: jax.Array
    root_w_local: jax.Array
    clip_id: jax.Array
    clip_start: jax.Array
    clip_end: jax.Array
    bin_clip: jax.Array
    bin_start: jax.Array
    frame_bin: jax.Array
    ref_contact: jax.Array
    names: tuple
    num_frames: int
    num_bins: int
    num_joints: int

    def goal_dim(self, K: int) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self.num_joints)

    def future_frames(self, f, K: int):
        """The K frames after f, held at the clip's last frame."""
        end = self.clip_end[self.clip_id[f]]
        fut = f[..., None] + jnp.arange(1, K + 1, dtype=f.dtype)
        return jnp.minimum(fut, end[..., None])

    def goal(self, f, K: int):
        """The policy's goal, from the reference only: K future frames relative to the current reference frame (..., K*(15+2J))."""
        fut = self.future_frames(f, K)
        Ht = tr3(yaw_mat(self.root_R[f]))[..., None, :, :]
        dp = mv3(Ht, self.root_p[fut] - self.root_p[f][..., None, :])
        rot = mat_to_6d(mm3(Ht, self.root_R[fut]))
        feats = jnp.concatenate((dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1)
        return feats.reshape(f.shape + (-1,))

    def ref_state(self, f):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def load_motion(motion_dir: str, kin, feet: list[int], margin: float = 0.015, vmax: float = 0.5, quant: float = 0.05) -> Motion:
    """All *.csv clips of ``motion_dir`` in sorted order (blinkmimic.motion.MotionLib.from_dir plus set_ref_contact(feet))."""
    manifest = Path(motion_dir) / "manifest.json"
    if manifest.is_file():
        rows = json.loads(manifest.read_text())["clips"]
        clips = []
        fields = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")
        for row in rows:
            path = manifest.parent / row["derived"]
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["derived_sha256"]:
                    raise ValueError(f"Reference hash mismatch: {path}")
                stream.seek(0)
                with np.load(stream, allow_pickle=False) as data:
                    p, quat, v, w, q, qd = (data[field] for field in fields)
            clips.append((p, _quat_to_mat_np(quat), v, w, q, qd))
        return assemble_motion(tuple(row["song"] for row in rows), clips, kin, feet, margin, vmax, quant)
    files = sorted(glob.glob(os.path.join(motion_dir, "*.csv")))
    assert files, f"no clips in {motion_dir}: run tools/download_data.sh first"
    names = tuple(os.path.splitext(os.path.basename(f))[0] for f in files)
    ps, Rs, vs, ws, js, jds = [], [], [], [], [], []
    for f in files:
        p, quat, j, fps = load_lafan1_csv(f)
        p, quat, j = (x.astype(np.float32) for x in (p, quat, j))
        T_src = p.shape[0]
        dur = (T_src - 1) / fps
        n = int(np.floor(dur / CONTROL_DT)) + 1
        t = np.arange(n, dtype=np.float32) * np.float32(CONTROL_DT) * np.float32(fps)
        i0 = np.minimum(np.floor(t).astype(np.int64), T_src - 2)
        a = np.clip(t - i0.astype(np.float32), 0, 1)
        p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
        j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
        quat = _slerp_np(quat[i0], quat[i0 + 1], a)
        R = _quat_to_mat_np(quat).astype(np.float32)
        w = np.zeros_like(p)
        w[1:-1] = _so3_log_np(R[2:] @ np.swapaxes(R[:-2], -1, -2)) / np.float32(2 * CONTROL_DT)
        w[0] = _so3_log_np(R[1] @ R[0].T) / np.float32(CONTROL_DT)
        w[-1] = _so3_log_np(R[-1] @ R[-2].T) / np.float32(CONTROL_DT)
        ps.append(p), Rs.append(R), vs.append(_gradient(p, CONTROL_DT)), ws.append(w), js.append(j), jds.append(_gradient(j, CONTROL_DT))
    clips = zip(ps, Rs, vs, ws, js, jds, strict=True)
    return assemble_motion(names, clips, kin, feet, margin, vmax, quant)


def assemble_motion(names, clips, kin, feet, margin, vmax, quant):
    """Build device arrays and per-clip reset/contact features from native states."""
    ps, Rs, vs, ws, js, jds = zip(*clips, strict=True)
    lens = np.asarray([len(p) for p in ps])
    clip_start = np.concatenate(([0], np.cumsum(lens[:-1]))).astype(np.int32)
    clip_end = (clip_start + lens - 1).astype(np.int32)
    clip_id = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(lens)])
    cat = lambda xs: jnp.asarray(np.concatenate(xs).astype(np.float32))  # noqa: E731
    root_p, root_R, root_v, root_w, q, qd = cat(ps), cat(Rs), cat(vs), cat(ws), cat(js), cat(jds)
    fk = jax.jit(kin.forward)
    out = [fk(root_p[s:s + 65536], root_R[s:s + 65536], root_v[s:s + 65536], root_w[s:s + 65536], q[s:s + 65536], qd[s:s + 65536])
           for s in range(0, len(clip_id), 65536)]  # fmt: skip
    body_p, body_R, body_v, body_w = (jnp.concatenate([o[i] for o in out]) for i in range(4))
    Rt = tr3(root_R)
    bin_clip, bin_start = [], []
    for c in range(len(lens)):
        for b in range(int(np.ceil(lens[c] / BIN_FRAMES))):
            bin_clip.append(c)
            bin_start.append(int(clip_start[c]) + b * BIN_FRAMES)
    frame_bin = np.zeros(len(clip_id), np.int32)
    for b, (c, s) in enumerate(zip(bin_clip, bin_start)):
        frame_bin[s : min(s + BIN_FRAMES, int(clip_end[c]) + 1)] = b
    # reference foot contact (BlinkMimic --ref_contact_fix): below the clip's own floor (quantile) + margin, slower than vmax
    z = np.asarray(body_p[:, feet, 2])
    vh = np.linalg.norm(np.asarray(body_v[:, feet, :2]), axis=-1)
    floor = np.zeros_like(z)
    for c in range(len(lens)):
        s = clip_id == c
        floor[s] = np.quantile(z[s], quant, axis=0)
    ref_contact = ((z < floor + margin) & (vh < vmax)).astype(np.float32)
    return Motion(root_p=root_p, root_R=root_R, root_v=root_v, root_w=root_w, q=q, qd=qd, body_p=body_p, body_R=body_R,
                  body_v=body_v, body_w=body_w, root_v_local=mv3(Rt, root_v), root_w_local=mv3(Rt, root_w),
                  clip_id=jnp.asarray(clip_id), clip_start=jnp.asarray(clip_start), clip_end=jnp.asarray(clip_end),
                  bin_clip=jnp.asarray(bin_clip, jnp.int32), bin_start=jnp.asarray(bin_start, jnp.int32),
                  frame_bin=jnp.asarray(frame_bin), ref_contact=jnp.asarray(ref_contact), names=names,
                  num_frames=int(len(clip_id)), num_bins=len(bin_start), num_joints=int(q.shape[-1]))  # fmt: skip
