"""Abstract base class for interactive tasks.

Users implement this interface to define task-specific logic
(reset, step, observation, termination) without touching the
domain-randomization pipeline.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from domain_rand.core.scene import Scene

if TYPE_CHECKING:
    from domain_rand.randomizers.base import DomainRandomizer


class Task(ABC):
    """Task interface for interactive demo collection.

    Subclass this and implement reset / step / get_observation.
    The DemoCollector calls these methods in a framework-managed loop.

    Lifecycle per episode:
        task.reset(scene, rng)  →  initial observation
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = task.step(scene, action)
    """

    # ── Required ─────────────────────────────────────────────────────────

    @abstractmethod
    def reset(self, scene: Scene, rng: np.random.Generator) -> dict:
        """Reset the task to a random initial state.

        Called once at the start of each episode, AFTER the domain
        randomization pass (texture/lighting/placement randomizers).

        The implementation should:
          - Randomise object / goal / robot initial states.
          - Set data.qpos / data.qvel / data.ctrl as needed.
          - Call scene.forward() or scene.step() to propagate changes.
          - Return the initial observation dict.

        Args:
            scene: The loaded Scene (model + data available).
            rng: Seeded numpy Generator for reproducibility.

        Returns:
            observation dict with at least {'state': np.ndarray}.
            May also contain 'rgb', 'depth', 'extrinsics', 'intrinsics'
            if the task pre-renders its own images; otherwise the
            DemoCollector handles rendering via the configured camera.
        """

    @abstractmethod
    def step(self, scene: Scene, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Execute one step of the task.

        The implementation should:
          - Apply the action (set ctrl / qpos).
          - Call scene.step() one or more times.
          - Compute reward and termination.
          - Return the new observation, reward, done flag, and info dict.

        Args:
            scene: The loaded Scene.
            action: Action vector (shape depends on task).

        Returns:
            (observation, reward, done, info)
        """

    @abstractmethod
    def get_observation(self, scene: Scene) -> dict:
        """Return the current observation dict.

        Must contain at least {'state': np.ndarray}.
        The DemoCollector will add rendered images to this dict.
        """

    # ── Optional ──────────────────────────────────────────────────────────

    def register_randomizers(
        self, dr: "DomainRandomizer", rng: np.random.Generator,
    ) -> None:
        """Register task-specific randomizers.

        Called once during DemoCollector initialization.  Override
        to add randomizers that are specific to your task — object
        placement, goal sampling, physics parameters, etc.

        The framework always registers texture, lighting, and camera
        randomizers automatically.  You only need to add the ones
        that are unique to your task.

        Example:
            from domain_rand.randomizers.placement import (
                ObjectPlacementRandomizer,
                PlacementRandomizationConfig,
            )
            cfg = PlacementRandomizationConfig(table_z=0.425, ...)
            dr.register("placement",
                ObjectPlacementRandomizer(cfg, rng),
                enabled=True,
            )
        """

    def get_done(self, scene: Scene) -> bool:
        """Check whether the episode should end (e.g. timeout, success).

        Override this if termination logic is separate from step().
        Default: always False (termination handled inside step() return).
        """
        return False

    @property
    def action_spec(self) -> dict:
        """Return metadata about the action space.

        Override in subclasses.  Used by the recorder for dataset metadata.

        Example:
            {'shape': (3,), 'dtype': 'float32', 'names': ['dx', 'dy', 'dyaw']}
        """
        return {"shape": (0,), "dtype": "float32"}

    @property
    def state_spec(self) -> dict:
        """Return metadata about the state vector.

        Override in subclasses.

        Example:
            {'shape': (6,), 'dtype': 'float32', 'names': ['jx','jy','jyaw','ox','oy','oyaw']}
        """
        return {"shape": (0,), "dtype": "float32"}
