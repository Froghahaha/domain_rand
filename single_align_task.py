"""Single Part Align Task — camera-to-object relative pose alignment.

Camera body is a static body (no joint).  Pose is set directly via
model.body_pos / model.body_quat — clean, no physics, no qpos wrangling.

Action: 6-DOF delta  [dx, dy, dz, droll, dpitch, dyaw]  (camera-frame)
State (17D):  [cam_pos(3), cam_quat_wxyz(4), tgt_pos(3), tgt_quat_wxyz(4), obj_pos(3)]
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from domain_rand.tasks.base import Task


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _look_at_rotation(cam_pos: np.ndarray, target_pos: np.ndarray) -> R:
    """Rotation that makes the camera look at target_pos.

    MuJoCo cameras look along -Z.  World up is +Z.
    """
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


class SinglePartAlign(Task):
    """Camera-to-object relative pose alignment."""

    def __init__(self, **kwargs):
        self.obj_body = "obj_1"
        self.cam_body = "cam_body"
        self.max_steps = 100
        self.table_z = 0.425
        self.obj_mesh_half_z = 0.043

        # ── Target pose (tight range around object) ─────────────
        self.target_dist_range = (0.35, 0.45)
        self.target_pos_offset_range = 0.1
        self.target_ang_offset = 5.0
        self.fixed_target = False          # True → same target across all episodes

        # ── Initial = target + perturbation ─────────────────────
        self.init_pos_perturb = 0.15    # max position perturbation (m)
        self.init_ang_perturb = 30.0    # max angular perturbation (deg)
        self.init_rz_perturb = 180.0    # max camera-Z rotation (deg, large — object stays centred)

        # ── Termination ─────────────────────────────────────────
        self.pos_threshold = 0.02
        self.ang_threshold = 0.05

        # ── Step limits ─────────────────────────────────────────
        self.max_linear_vel = 0.03
        self.max_angular_vel = 0.1

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self._cam_body_id = -1
        self._obj_body_id = -1
        self._target_pos = np.zeros(3)
        self._target_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    # ── Task interface ──────────────────────────────────────────

    def reset(self, scene, rng):
        if self._cam_body_id < 0:
            self._cam_body_id = scene.get_body_index(self.cam_body)
            self._obj_body_id = scene.get_body_index(self.obj_body)

        # 1. Place object on table
        obj_x = rng.uniform(-0.20, 0.20)*0
        obj_y = rng.uniform(-0.15, 0.15)*0
        obj_yaw = rng.uniform(-np.pi, np.pi)*0
        qw, qz = np.cos(obj_yaw / 2), np.sin(obj_yaw / 2)
        scene.model.body_pos[self._obj_body_id] = np.array(
            [obj_x, obj_y, self.table_z + self.obj_mesh_half_z])+rng.uniform(-0.2, 0.2, 3)*0
        scene.model.body_quat[self._obj_body_id] = np.array([qw, 0.0, 0.0, qz])
        scene.forward()

        obj_pos, _ = scene.get_body_pose(self.obj_body)

        # 2. Sample target camera pose (once if fixed, otherwise per-episode)
        self.fixed_target =True
        if not self.fixed_target or self._target_pos.sum() == 0.0:
            self._target_pos, self._target_quat_wxyz = self._sample_pose(
                rng, obj_pos, self.target_dist_range,
                self.target_pos_offset_range, self.target_ang_offset,
            )

        # 3. Initial = target + random perturbation
        init_pos = self._target_pos + rng.uniform(
            -self.init_pos_perturb, self.init_pos_perturb, 3)
        tgt_rot_xyzw = _wxyz_to_xyzw(self._target_quat_wxyz)
        perturb_angles = rng.uniform(
            -self.init_ang_perturb, self.init_ang_perturb, 3)
        perturb_rot = R.from_euler('xyz', perturb_angles, degrees=True)
        # Extra camera-Z rotation (large range — object stays centred)
        rz = rng.uniform(-self.init_rz_perturb, self.init_rz_perturb)
        rz_rot = R.from_euler('z', rz, degrees=True)
        init_rot = perturb_rot * R.from_quat(tgt_rot_xyzw) * rz_rot
        init_quat = _xyzw_to_wxyz(init_rot.as_quat())

        # 4. Set camera body pose directly
        scene.model.body_pos[self._cam_body_id] = init_pos
        scene.model.body_quat[self._cam_body_id] = init_quat
        scene.forward()

        self._step_count = 0
        # self._debug_viewer = True  
        # --- Debug viewer ---
        if getattr(self, '_debug_viewer', False):
            import mujoco
            import mujoco.viewer
            with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
                # === 开启坐标系可视化 ===
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
                
                # === 增大坐标系尺寸 ===
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # 世界坐标系参考
                viewer.opt.frame = 1  # 增大坐标系线宽（0=关闭, 1=小, 2=中, 3=大）
                
                # 或者更直接地设置坐标系缩放
                # 注意：有些版本通过 scale 控制
                # viewer.opt.scale = 1.0  # 整体缩放
                
                # 降低透明度背景以便看清
                viewer.opt.geomgroup[0] = 1  # 显示所有几何体
                while viewer.is_running():
                    viewer.sync()

        return self.get_observation(scene)

    def step(self, scene, action):
        action = np.asarray(action, dtype=np.float64).ravel()

        cam_pos = scene.model.body_pos[self._cam_body_id].copy()
        cam_quat = scene.model.body_quat[self._cam_body_id].copy()
        cam_rot = R.from_quat(_wxyz_to_xyzw(cam_quat))
        R_cam = cam_rot.as_matrix()  # camera → world

        # Convert camera-frame deltas → world frame
        dp_cam = np.clip(action[:3], -self.max_linear_vel, self.max_linear_vel)
        dp_world = R_cam @ dp_cam
        new_pos = cam_pos + dp_world

        ang_cam = np.clip(action[3:], -self.max_angular_vel, self.max_angular_vel)
        ang_world = R_cam @ ang_cam
        new_rot = R.from_rotvec(ang_world) * cam_rot
        new_quat = _xyzw_to_wxyz(new_rot.as_quat())

        # Write directly to body pose
        scene.model.body_pos[self._cam_body_id] = new_pos
        scene.model.body_quat[self._cam_body_id] = new_quat
        scene.forward()

        pos_err = np.linalg.norm(self._target_pos - new_pos)
        tgt_rot = R.from_quat(_wxyz_to_xyzw(self._target_quat_wxyz))
        ang_err = np.linalg.norm((tgt_rot * new_rot.inv()).as_rotvec())

        reward = -(pos_err + 0.1 * ang_err)
        self._step_count += 1
        done = (
            (pos_err < self.pos_threshold and ang_err < self.ang_threshold)
            or (self._step_count >= self.max_steps)
        )

        obs = self.get_observation(scene)
        info = {
            "pos_error": float(pos_err),
            "ang_error": float(ang_err),
            "success": bool(pos_err < self.pos_threshold and ang_err < self.ang_threshold),
        }
        return obs, reward, done, info

    def get_observation(self, scene):
        cam_pos = scene.model.body_pos[self._cam_body_id].copy()
        cam_quat = scene.model.body_quat[self._cam_body_id].copy()
        obj_pos = scene.model.body_pos[self._obj_body_id].copy()
        state = np.concatenate([cam_pos, cam_quat,
                                self._target_pos, self._target_quat_wxyz,
                                obj_pos])
        return {"state": state.astype(np.float32)}

    @property
    def action_spec(self):
        return {"shape": (6,), "dtype": "float32",
                "names": ["vx", "vy", "vz", "wx", "wy", "wz"]}

    @property
    def state_spec(self):
        return {"shape": (17,), "dtype": "float32"}

    # ── Pose sampling ────────────────────────────────────────────

    def _sample_pose(self, rng, obj_pos, dist_range, pos_offset_range, ang_offset_deg):
        # Random position on sphere around object
        dist = rng.uniform(*dist_range)
        v = rng.normal(0, 1, 3)
        v = v / np.linalg.norm(v)
        cam_pos = obj_pos + v * dist
        if cam_pos[2] < self.table_z + 0.05:
            cam_pos[2] = self.table_z + 0.05 + abs(v[2]) * dist * 0.5

        cam_pos += rng.uniform(-pos_offset_range, pos_offset_range, 3)

        # Look-at + orientation offset
        look_rot = _look_at_rotation(cam_pos, obj_pos)
        angles = rng.uniform(-ang_offset_deg, ang_offset_deg, 3)
        offset_rot = R.from_euler('xyz', angles, degrees=True)
        final_rot = look_rot*offset_rot

        return cam_pos, _xyzw_to_wxyz(final_rot.as_quat())
