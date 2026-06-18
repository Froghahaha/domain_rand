"""Scripted geodesic policy with feedforward orientation compensation.

Position:  radial (distance) + tangential (sphere surface) geodesic.

Orientation:
  1. Feedforward — pre-compensate the rotation that the translation *would* cause
     (so the object stays centred despite the camera moving).
  2. Feedback    — PD toward the target quaternion (closes the remaining gap).

State (17D):
  [cam_pos(3), cam_quat_wxyz(4), tgt_pos(3), tgt_quat_wxyz(4), obj_pos(3)]
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from domain_rand.policy.base import Policy


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def _look_at_rotation(cam_pos: np.ndarray, target_pos: np.ndarray) -> R:
    """Rotation that points the camera at target_pos (MuJoCo: looks along -Z)."""
    forward = target_pos - cam_pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / np.linalg.norm(right)
    true_up = np.cross(forward, right)

    rot_mat = np.column_stack([right, true_up, -forward])
    if np.linalg.det(rot_mat) < 0:
        rot_mat[:, 0] = -rot_mat[:, 0]
    return R.from_matrix(rot_mat)


class ScriptedAlignPolicy(Policy):
    """Geodesic position + feedforward-compensated orientation."""

    def __init__(
        self,
        kp_radial: float = 0.3,
        kp_tangent: float = 0.18,
        kp_ang: float = 0.15,
        max_vel: float = 0.015,
        max_ang: float = 0.04,
    ):
        self.kp_radial = kp_radial
        self.kp_tangent = kp_tangent
        self.kp_ang = kp_ang
        self.max_vel = max_vel
        self.max_ang = max_ang

    def get_action(self, observation: dict) -> np.ndarray:
        state = observation["state"]  # (17,)

        cam_pos = state[0:3]
        cam_quat_wxyz = state[3:7]
        tgt_pos = state[7:10]
        tgt_quat_wxyz = state[10:14]
        obj_pos = state[14:17]

        # ═══════════════════════════════════════════════════════════
        #  Position — geodesic (radial + tangential)
        # ═══════════════════════════════════════════════════════════

        r_curr = cam_pos - obj_pos
        d_curr = np.linalg.norm(r_curr)
        if d_curr < 1e-8:
            d_curr = 1e-8
        u_curr = r_curr / d_curr

        r_tgt = tgt_pos - obj_pos
        d_tgt = np.linalg.norm(r_tgt)
        if d_tgt < 1e-8:
            d_tgt = 1e-8

        # Radial
        v_radial = self.kp_radial * (d_tgt - d_curr) * u_curr

        # Tangential
        u_tgt = r_tgt / d_tgt
        tangent = u_tgt - np.dot(u_tgt, u_curr) * u_curr
        tan_norm = np.linalg.norm(tangent)
        if tan_norm > 1e-8:
            tangent = tangent / tan_norm
        cos_a = np.clip(np.dot(u_curr, u_tgt), -1.0, 1.0)
        v_tangential = self.kp_tangent * np.arccos(cos_a) * d_curr * tangent

        pos_action = v_radial + v_tangential
        dp = np.clip(pos_action, -self.max_vel, self.max_vel)

        # ═══════════════════════════════════════════════════════════
        #  Orientation — feedforward compensation + feedback PD
        # ═══════════════════════════════════════════════════════════

        cam_rot = R.from_quat(_wxyz_to_xyzw(cam_quat_wxyz))
        tgt_rot = R.from_quat(_wxyz_to_xyzw(tgt_quat_wxyz))

        # Feedforward: the rotation change caused by the position delta dp.
        # If we move the camera by dp, the look-at rotation changes from
        # R_before to R_after.  We pre-apply that delta so the object stays
        # centred.
        look_before = _look_at_rotation(cam_pos, obj_pos)
        look_after = _look_at_rotation(cam_pos + dp, obj_pos)
        ff_delta = look_after * look_before.inv()
        ang_ff = ff_delta.as_rotvec()

        # Feedback: PD toward the target quaternion (starting from the
        # feedforward-compensated orientation).
        compensated_rot = ff_delta * cam_rot
        ang_fb = (tgt_rot * compensated_rot.inv()).as_rotvec()
        ang_fb = self.kp_ang * ang_fb

        ang_world = ang_ff + ang_fb

        # ═══════════════════════════════════════════════════════════
        #  Convert world → camera frame, clamp, return
        # ═══════════════════════════════════════════════════════════

        R_cam_inv = cam_rot.inv()
        dp_cam = R_cam_inv.apply(dp)
        ang_cam = R_cam_inv.apply(ang_world)

        dp_cam = np.clip(dp_cam, -self.max_vel, self.max_vel)
        ang_cam = np.clip(ang_cam, -self.max_ang, self.max_ang)
        action = np.concatenate([dp_cam, ang_cam])
        return action.astype(np.float32)

    def reset(self) -> None:
        pass
