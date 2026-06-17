"""Texture and color randomization for MuJoCo geoms.

Modifies model.geom_rgba values per-geom with configurable modes:
  - rgb: random uniform RGB per geom
  - checker: random color pairs for checker-like appearance
  - noise: random single color per geom (simple noise approximation)
  - gradient: random single color per geom (simple gradient approximation)
"""

import mujoco
import numpy as np

from domain_rand.core.config import TextureRandomizationConfig
from domain_rand.randomizers.base import Randomizer


class TextureRandomizer(Randomizer):
    """Randomizes geom colors/textures in the MuJoCo scene."""

    def __init__(self, config: TextureRandomizationConfig, rng: np.random.Generator):
        super().__init__(rng)
        self.config = config
        self._nominal_rgba: np.ndarray | None = None
        self._nominal_matid: np.ndarray | None = None

    def save_nominal(self, model: mujoco.MjModel) -> None:
        self._nominal_rgba = model.geom_rgba.copy()
        self._nominal_matid = model.geom_matid.copy()

    def randomize(self, model: mujoco.MjModel) -> None:
        if not self.config.enabled:
            return

        rmin, rmax, gmin, gmax, bmin, bmax, amin, amax = self.config.rgba_range

        for i in range(model.ngeom):
            # Skip excluded geoms
            if self._is_excluded(model, i):
                continue

            if self.config.mode == "rgb":
                model.geom_rgba[i, 0] = self.rng.uniform(rmin, rmax)
                model.geom_rgba[i, 1] = self.rng.uniform(gmin, gmax)
                model.geom_rgba[i, 2] = self.rng.uniform(bmin, bmax)
                model.geom_rgba[i, 3] = self.rng.uniform(amin, amax)

            elif self.config.mode == "checker":
                # Simulate checker by assigning one of two random color pairs
                pair = self.rng.integers(0, 2)
                if pair == 0:
                    model.geom_rgba[i, :3] = self.rng.uniform(rmin, rmax, 3)
                else:
                    model.geom_rgba[i, :3] = self.rng.uniform(rmin, rmax, 3)
                model.geom_rgba[i, 3] = 1.0

            elif self.config.mode in ("noise", "gradient"):
                # For procedural textures, randomize the base color
                model.geom_rgba[i, :3] = self.rng.uniform(rmin, rmax, 3)
                model.geom_rgba[i, 3] = 1.0

    def restore(self, model: mujoco.MjModel) -> None:
        if self._nominal_rgba is not None:
            model.geom_rgba[:] = self._nominal_rgba
            model.geom_matid[:] = self._nominal_matid

    def get_state(self, model: mujoco.MjModel) -> dict:
        """Return current rgba values for recording."""
        return {
            "geom_rgba": model.geom_rgba.copy(),
            "mode": self.config.mode,
        }

    def _is_excluded(self, model: mujoco.MjModel, geom_idx: int) -> bool:
        """Check if a geom should be excluded from randomization."""
        addr = model.name_geomadr[geom_idx]
        if addr < 0:
            return False
        name = model.names[addr:].split(b"\x00")[0].decode()
        for excl in self.config.exclude_geoms:
            if excl in name:
                return True
        return False
