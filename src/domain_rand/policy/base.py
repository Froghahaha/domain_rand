"""Abstract base class for action-generation policies.

Teleoperation, scripted controllers, and learned models all implement
this single-method interface so they can be swapped without changing
the demo collection pipeline.
"""

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """Action-generation interface.

    A Policy maps an observation dict to an action vector (or None,
    meaning "skip this frame" — useful for teleop when no key is held).

    Lifecycle:
        policy.reset()               # called once per episode
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, _ = task.step(scene, action)
    """

    @abstractmethod
    def get_action(self, observation: dict) -> np.ndarray | None:
        """Return the action to take given the current observation.

        Args:
            observation: Dict from Task.get_observation() augmented
                         with images rendered by the DemoCollector.

        Returns:
            Action vector as a float32 numpy array, or None to skip.
        """

    def reset(self) -> None:
        """Called at the start of each episode. Optional hook."""
        pass

    def close(self) -> None:
        """Called when demo collection ends. Optional cleanup hook."""
        pass
