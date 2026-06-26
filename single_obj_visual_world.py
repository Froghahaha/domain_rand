"""Single Object Visual World Task — camera trajectory on sphere around object.

Generates a dense viewpoint library (~10000 points) uniformly distributed on
the unit sphere centered at the object.  Each demo picks two random viewpoints
as start/end, applies radial/tangential/angular perturbations, and the camera
moves along a geodesic (great-circle) path with adaptive step count so that
every step respects the velocity limits.

Camera body is a static body (no joint).  Pose is set directly via
model.body_pos / model.body_quat.

State (10D):  [cam_pos(3), cam_quat_wxyz(4), obj_pos(3)]
Action (6D):  [dx, dy, dz, droll, dpitch, dyaw]  (camera-frame)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from domain_rand.tasks.base import Task
from domain_rand.policy.base import Policy


# ═══════════════════════════════════════════════════════════════════════════
#  Quaternion conversion helpers
# ═══════════════════════════════════════════════════════════════════════════

def _wxyz_to_xyzw(q):
    """MuJoCo [w,x,y,z] → scipy [x,y,z,w]."""
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q):
    """scipy [x,y,z,w] → MuJoCo [w,x,y,z]."""
    return np.array([q[3], q[0], q[1], q[2]])


# ═══════════════════════════════════════════════════════════════════════════
#  Camera look-at — proven reliable from single_align_task.py
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  Fibonacci sphere — uniform sampling on unit sphere
# ═══════════════════════════════════════════════════════════════════════════

def _sample_fibonacci_sphere(n: int) -> np.ndarray:
    """Uniformly sample n points on the unit sphere (Fibonacci lattice).

    Returns (n, 3) float64 array of unit vectors.  Much more uniform than
    normalised-Gaussian sampling — no clustering at the poles.
    """
    points = np.empty((n, 3), dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0  # 1 → -1
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        points[i, 0] = np.cos(theta) * radius
        points[i, 1] = np.sin(theta) * radius
        points[i, 2] = y
    return points


# ═══════════════════════════════════════════════════════════════════════════
#  Tangent plane helpers
# ═══════════════════════════════════════════════════════════════════════════

def _tangent_basis(d: np.ndarray):
    """Return two orthonormal vectors {u, v} spanning the tangent plane at d.

    d must be a unit vector.  u, v are perpendicular to d and to each other.
    """
    if abs(d[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, d)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


# ═══════════════════════════════════════════════════════════════════════════
#  SLERP for direction vectors
# ═══════════════════════════════════════════════════════════════════════════

def _slerp_vectors(v1: np.ndarray, v2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors.

    Args:
        v1, v2: (3,) unit vectors.
        t: (k,) array of interpolation parameters in [0, 1].

    Returns:
        (k, 3) array of interpolated unit vectors.
    """
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.tile(v1, (len(t), 1))
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a[:, None] * v1 + b[:, None] * v2


# ═══════════════════════════════════════════════════════════════════════════
#  Geodesic waypoint computation
# ═══════════════════════════════════════════════════════════════════════════

def _compute_geodesic_waypoints(
    pos_start: np.ndarray,
    pos_end: np.ndarray,
    quat_start: np.ndarray,  # wxyz
    quat_end: np.ndarray,    # wxyz
    obj_center: np.ndarray,
    n_steps: int,
) -> list:
    """Compute *n_steps* waypoints along a geodesic from start to end.

    Direction vectors (from obj_center) are SLERP-interpolated; radii are
    linearly interpolated; orientation perturbations (offset from pure
    look-at) are SLERP-interpolated via scipy.

    Returns:
        List of (pos, quat_wxyz) tuples, length *n_steps*.
    """
    # ── Directions & radii ──────────────────────────────────────
    d_start_vec = pos_start - obj_center
    d_end_vec = pos_end - obj_center
    r_start = np.linalg.norm(d_start_vec)
    r_end = np.linalg.norm(d_end_vec)
    d_start = d_start_vec / r_start
    d_end = d_end_vec / r_end

    # ── Decompose orientation into look-at + perturbation ───────
    R_start = R.from_quat(_wxyz_to_xyzw(quat_start))
    R_end = R.from_quat(_wxyz_to_xyzw(quat_end))
    R_look_start = _look_at_rotation(pos_start, obj_center)
    R_look_end = _look_at_rotation(pos_end, obj_center)
    R_perturb_start = R_look_start.inv() * R_start
    R_perturb_end = R_look_end.inv() * R_end

    # ── Interpolation parameters ────────────────────────────────
    t = np.linspace(0.0, 1.0, n_steps)

    # SLERP directions
    dirs = _slerp_vectors(d_start, d_end, t)                     # (n_steps, 3)

    # Linear radii
    radii = r_start + t * (r_end - r_start)                      # (n_steps,)

    # SLERP perturbation via scipy
    pert_slerp_obj = Slerp([0.0, 1.0], R.concatenate([R_perturb_start, R_perturb_end]))
    R_perturb_all = pert_slerp_obj(t)                             # Rotation with n_steps elements

    # ── Build waypoints ─────────────────────────────────────────
    waypoints = []
    for i in range(n_steps):
        pos_i = obj_center + radii[i] * dirs[i]
        R_look_i = _look_at_rotation(pos_i, obj_center)
        R_i = R_look_i * R_perturb_all[i]
        quat_i = _xyzw_to_wxyz(R_i.as_quat())
        waypoints.append((pos_i, quat_i))

    return waypoints


# ═══════════════════════════════════════════════════════════════════════════
#  Task
# ═══════════════════════════════════════════════════════════════════════════

class SingleObjVisualWorld(Task):
    """Camera-trajectory task over a viewpoint sphere around an object.

    Each episode: pick two random viewpoints from a ~10000-point Fibonacci
    library, apply perturbations, compute a geodesic path, and step the
    camera along it.  Step count is adaptive — determined by the arc length
    and the (randomly perturbed) velocity limits.
    """

    def __init__(self, **kwargs):
        # ── Object / camera identities ──────────────────────────
        self.obj_body = "obj_1"
        self.cam_body = "cam_body"

        # ── Viewpoint library ───────────────────────────────────
        self.num_viewpoints = 10000
        self.sphere_radius = 0.40         # base camera distance (m)

        # ── Per-viewpoint perturbations ─────────────────────────
        self.ang_perturb_deg = 8.0        # orientation jitter (degrees, per-axis)
        self.radial_perturb = 0.05        # radial distance jitter (m, ±)
        self.tangent_perturb = 0.03       # tangent-plane jitter (m, ±)

        # ── Velocity limits (base — per-demo randomisation below)
        self.max_linear_vel = 0.03        # max translation per step (m)
        self.max_angular_vel = 0.1        # max rotation per step (rad)
        self.vel_perturb = 0.5            # ±50% velocity randomisation per demo

        # ── Safety ──────────────────────────────────────────────
        self.max_steps = 500

        # ── Object placement ────────────────────────────────────
        self.table_z = 0.425
        self.obj_mesh_half_z = 0.043

        # kwargs override
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # ── Internal state ──────────────────────────────────────
        self._viewpoint_library = None    # built lazily on first reset
        self._waypoints: list = []        # [(pos, quat_wxyz), ...]
        self._waypoint_idx = 0
        self._obj_pos = np.zeros(3)
        self._cam_body_id = -1
        self._obj_body_id = -1

    # ── Task interface ──────────────────────────────────────────

    def reset(self, scene, rng):
        # Lazy body-index lookup
        if self._cam_body_id < 0:
            self._cam_body_id = scene.get_body_index(self.cam_body)
            self._obj_body_id = scene.get_body_index(self.obj_body)

        # 1. Place object on table (fixed at origin, same as original task)
        scene.model.body_pos[self._obj_body_id] = np.array(
            [0.0, 0.0, self.table_z + self.obj_mesh_half_z])
        scene.model.body_quat[self._obj_body_id] = np.array(
            [1.0, 0.0, 0.0, 0.0])
        scene.forward()
        self._obj_pos, _ = scene.get_body_pose(self.obj_body)

        # 2. Lazy-build viewpoint library
        if self._viewpoint_library is None:
            self._viewpoint_library = _sample_fibonacci_sphere(
                self.num_viewpoints)

        # 3. Per-demo velocity randomisation
        eff_lin_vel = self.max_linear_vel * rng.uniform(
            1.0 - self.vel_perturb, 1.0 + self.vel_perturb)
        eff_ang_vel = self.max_angular_vel * rng.uniform(
            1.0 - self.vel_perturb, 1.0 + self.vel_perturb)

        # 4. Pick two distinct viewpoints from library
        idx = rng.choice(self.num_viewpoints, size=2, replace=False)
        d_lib_start = self._viewpoint_library[idx[0]]
        d_lib_end = self._viewpoint_library[idx[1]]

        # 5. Apply perturbations to start & end viewpoints
        pos_start, quat_start = self._perturb_viewpoint(
            rng, d_lib_start, self._obj_pos)
        pos_end, quat_end = self._perturb_viewpoint(
            rng, d_lib_end, self._obj_pos)

        # 6. Compute geodesic arc parameters
        d_s = pos_start - self._obj_pos
        d_e = pos_end - self._obj_pos
        r_s = np.linalg.norm(d_s)
        r_e = np.linalg.norm(d_e)
        d_s_u = d_s / r_s
        d_e_u = d_e / r_e

        cos_theta = np.clip(np.dot(d_s_u, d_e_u), -1.0, 1.0)
        theta = np.arccos(cos_theta)                     # geodesic angle [0, π]
        avg_r = 0.5 * (r_s + r_e)
        arc_len = theta * avg_r                          # approx arc length (m)

        # 7. Determine step count from velocity limits
        n_pos = int(np.ceil(arc_len / max(eff_lin_vel, 1e-8)))
        n_ang = int(np.ceil(theta / max(eff_ang_vel, 1e-8)))
        n = max(n_pos, n_ang, 2)
        n = min(n, self.max_steps)

        # 8. Compute waypoints
        self._waypoints = _compute_geodesic_waypoints(
            pos_start, pos_end, quat_start, quat_end,
            self._obj_pos, n,
        )
        self._waypoint_idx = 0

        # 9. Set camera to first waypoint
        scene.model.body_pos[self._cam_body_id] = self._waypoints[0][0].copy()
        scene.model.body_quat[self._cam_body_id] = self._waypoints[0][1].copy()
        scene.forward()

        self._step_count = 0

        return self.get_observation(scene)

    def step(self, scene, action):
        """Advance one step along the pre-computed geodesic path.

        The *action* parameter is **ignored** — the task is self-driving.
        The companion ``VisualWorldPolicy`` returns the correct action for
        recording purposes.
        """
        # Advance waypoint index (clamp at last)
        if self._waypoint_idx < len(self._waypoints) - 1:
            self._waypoint_idx += 1

        # Set camera body pose
        pos, quat = self._waypoints[self._waypoint_idx]
        scene.model.body_pos[self._cam_body_id] = pos.copy()
        scene.model.body_quat[self._cam_body_id] = quat.copy()
        scene.forward()

        self._step_count += 1

        done = (self._waypoint_idx >= len(self._waypoints) - 1) or \
               (self._step_count >= self.max_steps)

        reward = 0.0  # trajectory recording, not RL

        obs = self.get_observation(scene)
        info = {
            "waypoint_idx": self._waypoint_idx,
            "total_waypoints": len(self._waypoints),
            "progress": float(self._waypoint_idx / max(len(self._waypoints) - 1, 1)),
        }
        return obs, reward, done, info

    def get_observation(self, scene):
        cam_pos = scene.model.body_pos[self._cam_body_id].copy()
        cam_quat = scene.model.body_quat[self._cam_body_id].copy()

        state = np.concatenate([cam_pos, cam_quat, self._obj_pos])

        # Next target waypoint (for the policy to compute action toward)
        if len(self._waypoints) > 0:
            next_idx = min(self._waypoint_idx + 1, len(self._waypoints) - 1)
            target_pos, target_quat = self._waypoints[next_idx]
        else:
            target_pos = cam_pos.copy()
            target_quat = cam_quat.copy()

        return {
            "state": state.astype(np.float32),
            "target_pos": target_pos.astype(np.float32),
            "target_quat": target_quat.astype(np.float32),
        }

    @property
    def action_spec(self):
        return {
            "shape": (6,), "dtype": "float32",
            "names": ["vx", "vy", "vz", "wx", "wy", "wz"],
        }

    @property
    def state_spec(self):
        return {"shape": (10,), "dtype": "float32"}

    # ── Internal helpers ────────────────────────────────────────

    def _perturb_viewpoint(self, rng, direction, obj_center):
        """Apply radial, tangential, and angular perturbations.

        Args:
            rng: numpy random Generator.
            direction: (3,) unit vector from the viewpoint library.
            obj_center: (3,) object world position.

        Returns:
            (pos, quat_wxyz) — world-frame camera pose.
        """
        # Radial perturbation
        r = self.sphere_radius + rng.uniform(
            -self.radial_perturb, self.radial_perturb)
        r = max(r, 0.05)  # minimum distance from object surface

        # Tangential (tangent-plane) perturbation
        u, v = _tangent_basis(direction)
        tangent_offset = (
            rng.uniform(-self.tangent_perturb, self.tangent_perturb) * u +
            rng.uniform(-self.tangent_perturb, self.tangent_perturb) * v
        )

        # World position
        pos = obj_center + r * direction + tangent_offset

        # Keep camera above table
        if pos[2] < self.table_z + 0.05:
            pos[2] = self.table_z + 0.05

        # Look-at rotation + angular perturbation
        look_rot = _look_at_rotation(pos, obj_center)
        angles = rng.uniform(-self.ang_perturb_deg, self.ang_perturb_deg, 3)
        offset_rot = R.from_euler('xyz', angles, degrees=True)
        final_rot = look_rot * offset_rot

        return pos, _xyzw_to_wxyz(final_rot.as_quat())


# ═══════════════════════════════════════════════════════════════════════════
#  Companion policy — extracts target from observation, returns delta action
# ═══════════════════════════════════════════════════════════════════════════

class VisualWorldPolicy(Policy):
    """Policy that follows the pre-computed geodesic trajectory.

    Reads the next target pose from the observation (set by the task) and
    computes the 6-DOF camera-frame delta action to reach it.  The task is
    self-driving (``step()`` ignores the action), but this policy ensures
    the actions recorded in the HDF5 dataset are meaningful.
    """

    def get_action(self, observation: dict) -> np.ndarray:
        state = observation["state"]
        cam_pos = state[0:3]
        cam_quat_wxyz = state[3:7]

        target_pos = observation["target_pos"]
        target_quat_wxyz = observation["target_quat"]

        # Current / target rotations
        cam_rot = R.from_quat(_wxyz_to_xyzw(cam_quat_wxyz))
        target_rot = R.from_quat(_wxyz_to_xyzw(target_quat_wxyz))

        # Position delta (world frame → camera frame)
        dp_world = target_pos - cam_pos

        # Orientation delta (world frame → camera frame)
        delta_rot = target_rot * cam_rot.inv()
        ang_world = delta_rot.as_rotvec()

        # Convert to camera frame
        R_cam_inv = cam_rot.inv()
        dp_cam = R_cam_inv.apply(dp_world)
        ang_cam = R_cam_inv.apply(ang_world)

        action = np.concatenate([dp_cam, ang_cam])
        return action.astype(np.float32)

    def reset(self) -> None:
        pass
