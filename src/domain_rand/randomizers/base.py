"""Abstract base class for all randomizers."""

from abc import ABC, abstractmethod

import mujoco
import numpy as np


class Randomizer(ABC):
    """Base class for all domain randomizers.

    Each randomizer operates on a MuJoCo model (mjModel) and is responsible
    for:
    1. Saving the nominal state before randomization.
    2. Applying randomized values to model attributes.
    3. Restoring the nominal state when needed.

    Subclasses must implement save_nominal, randomize, and restore.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    @abstractmethod
    def save_nominal(self, model: mujoco.MjModel) -> None:
        """Save the current model state so it can be restored later."""

    @abstractmethod
    def randomize(self, model: mujoco.MjModel) -> None:
        """Apply randomization to the model. Called once per episode."""

    @abstractmethod
    def restore(self, model: mujoco.MjModel) -> None:
        """Restore the model to its saved nominal state."""


class DomainRandomizer:
    """Composite randomizer that orchestrates multiple sub-randomizers.

    Runs them in registration order, with a configurable enabled flag
    to skip disabled randomizers.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._randomizers: list[tuple[str, Randomizer, bool]] = []

    def register(self, name: str, randomizer: Randomizer, enabled: bool = True) -> None:
        """Register a named randomizer.

        Args:
            name: Human-readable name for logging/debugging.
            randomizer: The Randomizer instance.
            enabled: Whether this randomizer is active.
        """
        self._randomizers.append((name, randomizer, enabled))

    def save_nominal(self, model: mujoco.MjModel) -> None:
        """Save nominal state across all registered randomizers."""
        for _, r, _ in self._randomizers:
            r.save_nominal(model)

    def randomize(self, model: mujoco.MjModel) -> dict[str, bool]:
        """Run all enabled randomizers.

        Returns:
            Dict mapping randomizer name -> whether it ran.
        """
        results = {}
        for name, r, enabled in self._randomizers:
            if enabled:
                r.randomize(model)
                results[name] = True
            else:
                results[name] = False
        return results

    def restore(self, model: mujoco.MjModel) -> None:
        """Restore nominal state for all randomizers."""
        for _, r, _ in self._randomizers:
            r.restore(model)

    def get_state(self, model: mujoco.MjModel) -> dict:
        """Collect the current randomized state from all sub-randomizers.

        Useful for recording the exact parameters used in each frame.
        """
        state = {}
        for name, r, enabled in self._randomizers:
            if hasattr(r, "get_state"):
                state[name] = r.get_state(model)
        return state
