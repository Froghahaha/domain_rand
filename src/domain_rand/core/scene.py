"""Scene management — loading, MjSpec operations, and nominal state tracking.

Provides a lightweight wrapper around the official MuJoCo Python bindings
for loading scenes, tracking nominal model state, and restoring it after
randomization.
"""

from pathlib import Path

import mujoco
import numpy as np


class Scene:
    """Manages a MuJoCo scene: loads the model, tracks nominal state.

    Usage:
        scene = Scene("path/to/scene.xml")
        scene.save_nominal()
        # ... randomize model attributes ...
        scene.forward()  # apply changes
        scene.restore()  # restore to nominal
    """

    def __init__(self, xml_path: str | Path):
        self.xml_path = Path(xml_path)

        # Load via MjSpec (MuJoCo 3.2.5+)
        self.spec = mujoco.MjSpec.from_file(str(self.xml_path))

        # Compile model and data
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        # Renderer cache
        self._renderer: mujoco.Renderer | None = None

        # Nominal state storage
        self._nominal: dict[str, np.ndarray] = {}

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def ngeom(self) -> int:
        return self.model.ngeom

    @property
    def nlight(self) -> int:
        return self.model.nlight

    @property
    def ncam(self) -> int:
        return self.model.ncam

    @property
    def camera_names(self) -> list[str]:
        """Return list of camera names in the scene."""
        names = []
        for i in range(self.model.ncam):
            addr = self.model.name_camadr[i]
            if addr >= 0:
                names.append(self.model.names[addr:].split(b"\x00")[0].decode())
            else:
                names.append(f"cam_{i}")
        return names

    @property
    def geom_names(self) -> list[str]:
        """Return list of geom names in the scene."""
        names = []
        for i in range(self.model.ngeom):
            addr = self.model.name_geomadr[i]
            if addr >= 0:
                names.append(self.model.names[addr:].split(b"\x00")[0].decode())
            else:
                names.append(f"geom_{i}")
        return names

    # ── Renderer ─────────────────────────────────────────────────────────

    def get_renderer(self, height: int = 480, width: int = 640) -> mujoco.Renderer:
        """Get or create a renderer with given resolution."""
        if self._renderer is None or self._renderer._height != height or self._renderer._width != width:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    # ── Simulation ───────────────────────────────────────────────────────

    def forward(self) -> None:
        """Run mj_forward to propagate state changes."""
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Run one simulation step (mujoco.mj_step)."""
        mujoco.mj_step(self.model, self.data)

    # ── Nominal state ────────────────────────────────────────────────────

    def save_nominal(self) -> None:
        """Save the current model state as 'nominal' for later restoration.

        Saves all modifiable visual attributes.
        """
        self._nominal = {
            "geom_rgba": self.model.geom_rgba.copy(),
            "geom_matid": self.model.geom_matid.copy(),
        }

        if self.model.nlight > 0:
            self._nominal["light_pos"] = self.model.light_pos.copy()
            self._nominal["light_dir"] = self.model.light_dir.copy()
            self._nominal["light_diffuse"] = self.model.light_diffuse.copy()
            self._nominal["light_ambient"] = self.model.light_ambient.copy()
            self._nominal["light_specular"] = self.model.light_specular.copy()
            self._nominal["light_type"] = self.model.light_type.copy()

        if self.model.ncam > 0:
            self._nominal["cam_pos"] = self.model.cam_pos.copy()
            self._nominal["cam_quat"] = self.model.cam_quat.copy()
            self._nominal["cam_fovy"] = self.model.cam_fovy.copy()

    def restore(self) -> None:
        """Restore the model to the saved nominal state."""
        if not self._nominal:
            return  # No nominal state saved yet

        self.model.geom_rgba[:] = self._nominal["geom_rgba"]
        self.model.geom_matid[:] = self._nominal["geom_matid"]

        if "light_pos" in self._nominal:
            self.model.light_pos[:] = self._nominal["light_pos"]
            self.model.light_dir[:] = self._nominal["light_dir"]
            self.model.light_diffuse[:] = self._nominal["light_diffuse"]
            self.model.light_ambient[:] = self._nominal["light_ambient"]
            self.model.light_specular[:] = self._nominal["light_specular"]
            self.model.light_type[:] = self._nominal["light_type"]

        if "cam_pos" in self._nominal:
            self.model.cam_pos[:] = self._nominal["cam_pos"]
            self.model.cam_quat[:] = self._nominal["cam_quat"]
            self.model.cam_fovy[:] = self._nominal["cam_fovy"]

        mujoco.mj_forward(self.model, self.data)

    # ── Convenience ──────────────────────────────────────────────────────

    def get_geom_index(self, name: str) -> int:
        """Get geom index by name. Returns -1 if not found."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        except Exception:
            return -1

    def get_camera_index(self, name: str) -> int:
        """Get camera index by name. Returns -1 if not found."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        except Exception:
            return -1

    def get_light_index(self, name: str) -> int:
        """Get light index by name. Returns -1 if not found."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, name)
        except Exception:
            return -1
