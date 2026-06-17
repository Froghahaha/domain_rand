"""Camera randomization for MuJoCo scenes.

Modifies camera position, orientation (quaternion), and field-of-view.
Supports multi-camera scenes with random camera selection.
"""

import mujoco
import numpy as np

from domain_rand.core.config import CameraRandomizationConfig
from domain_rand.randomizers.base import Randomizer


class CameraRandomizer(Randomizer):
    """Randomizes camera parameters in the MuJoCo scene."""

    def __init__(self, config: CameraRandomizationConfig, rng: np.random.Generator):
        super().__init__(rng)
        self.config = config
        self._nominal: dict[str, np.ndarray] = {}
        self._active_camera: int = 0

    def save_nominal(self, model: mujoco.MjModel) -> None:
        if model.ncam == 0:
            return
        self._nominal = {
            "cam_pos": model.cam_pos.copy(),
            "cam_quat": model.cam_quat.copy(),
            "cam_fovy": model.cam_fovy.copy(),
        }

    def randomize(self, model: mujoco.MjModel) -> None:
        if not self.config.enabled or model.ncam == 0:
            return

        ncam = model.ncam

        # Pick random camera if multiple and config says so
        if self.config.random_camera and ncam > 1:
            self._active_camera = self.rng.integers(0, ncam)

        cam_id = self._active_camera

        # Position jitter around nominal
        jx, jy, jz = self.config.position_jitter
        model.cam_pos[cam_id, 0] = self._nominal["cam_pos"][cam_id, 0] + self.rng.uniform(-jx, jx)
        model.cam_pos[cam_id, 1] = self._nominal["cam_pos"][cam_id, 1] + self.rng.uniform(-jy, jy)
        model.cam_pos[cam_id, 2] = self._nominal["cam_pos"][cam_id, 2] + self.rng.uniform(-jz, jz)

        # Rotation jitter: add small noise to quaternion, then renormalize
        quat_noise = self.rng.uniform(
            -self.config.rotation_jitter, self.config.rotation_jitter, size=4
        )
        model.cam_quat[cam_id, :] = self._nominal["cam_quat"][cam_id] + quat_noise
        norm = np.linalg.norm(model.cam_quat[cam_id])
        if norm > 1e-8:
            model.cam_quat[cam_id] /= norm

        # Field-of-view
        fov_min, fov_max = self.config.fovy_range
        model.cam_fovy[cam_id] = self.rng.uniform(fov_min, fov_max)

    def restore(self, model: mujoco.MjModel) -> None:
        if not self._nominal:
            return
        model.cam_pos[:] = self._nominal["cam_pos"]
        model.cam_quat[:] = self._nominal["cam_quat"]
        model.cam_fovy[:] = self._nominal["cam_fovy"]

    @property
    def active_camera(self) -> int:
        """The currently selected camera index (after last randomize call)."""
        return self._active_camera

    def get_state(self, model: mujoco.MjModel) -> dict:
        """Return current camera state for recording."""
        return {
            "cam_pos": model.cam_pos.copy(),
            "cam_quat": model.cam_quat.copy(),
            "cam_fovy": model.cam_fovy.copy(),
            "active_camera": self._active_camera,
        }
