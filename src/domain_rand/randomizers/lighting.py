"""Lighting randomization for MuJoCo scenes.

Modifies per-light attributes: position, direction, diffuse/ambient/specular
color, and optionally toggles lights on/off.
"""

import mujoco
import numpy as np

from domain_rand.core.config import LightingRandomizationConfig
from domain_rand.randomizers.base import Randomizer


class LightingRandomizer(Randomizer):
    """Randomizes lighting parameters in the MuJoCo scene."""

    def __init__(self, config: LightingRandomizationConfig, rng: np.random.Generator):
        super().__init__(rng)
        self.config = config
        # Nominal state storage
        self._nominal: dict[str, np.ndarray] = {}

    def save_nominal(self, model: mujoco.MjModel) -> None:
        if model.nlight == 0:
            return
        self._nominal = {
            "light_pos": model.light_pos.copy(),
            "light_dir": model.light_dir.copy(),
            "light_diffuse": model.light_diffuse.copy(),
            "light_ambient": model.light_ambient.copy(),
            "light_specular": model.light_specular.copy(),
            "light_type": model.light_type.copy(),
        }

    def randomize(self, model: mujoco.MjModel) -> None:
        if not self.config.enabled or model.nlight == 0:
            return

        nlight = model.nlight
        d_min, d_max = self.config.diffuse_range
        a_min, a_max = self.config.ambient_range
        s_min, s_max = self.config.specular_range

        for i in range(nlight):
            # Random toggle
            if self.config.random_toggle:
                is_on = self.rng.random() < self.config.toggle_probability
                if not is_on:
                    # Turn off light by setting all color components to zero
                    model.light_diffuse[i, :] = 0.0
                    model.light_ambient[i, :] = 0.0
                    model.light_specular[i, :] = 0.0
                    continue

            # Position jitter
            pos_jitter = self.rng.uniform(
                -self.config.position_jitter, self.config.position_jitter, size=3
            )
            model.light_pos[i, :] = self._nominal["light_pos"][i] + pos_jitter

            # Direction jitter (rotate by small angle)
            dir_jitter = self.rng.uniform(
                -self.config.direction_jitter, self.config.direction_jitter, size=3
            )
            model.light_dir[i, :] = self._nominal["light_dir"][i] + dir_jitter
            # Normalize direction
            norm = np.linalg.norm(model.light_dir[i])
            if norm > 1e-8:
                model.light_dir[i] /= norm

            # Color randomization
            model.light_diffuse[i, :] = self.rng.uniform(d_min, d_max, size=3)
            model.light_ambient[i, :] = self.rng.uniform(a_min, a_max, size=3)
            model.light_specular[i, :] = self.rng.uniform(s_min, s_max, size=3)

    def restore(self, model: mujoco.MjModel) -> None:
        if not self._nominal:
            return
        model.light_pos[:] = self._nominal["light_pos"]
        model.light_dir[:] = self._nominal["light_dir"]
        model.light_diffuse[:] = self._nominal["light_diffuse"]
        model.light_ambient[:] = self._nominal["light_ambient"]
        model.light_specular[:] = self._nominal["light_specular"]
        model.light_type[:] = self._nominal["light_type"]

    def get_state(self, model: mujoco.MjModel) -> dict:
        """Return current lighting state for recording."""
        return {
            "light_pos": model.light_pos.copy(),
            "light_dir": model.light_dir.copy(),
            "light_diffuse": model.light_diffuse.copy(),
            "light_ambient": model.light_ambient.copy(),
            "light_specular": model.light_specular.copy(),
        }
