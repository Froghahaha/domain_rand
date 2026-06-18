"""Abstract base class for IL trajectory recorders.

Implement this to support custom HDF5 output formats.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseRecorder(ABC):
    """Interface for pluggable HDF5 trajectory recorders.

    Lifecycle:
        recorder = MyRecorder(output_path)
        recorder.open(attrs)
        for ep in range(N):
            recorder.write_episode(ep, observations, actions, rewards, dones, infos, meta)
        recorder.close()
    """

    @abstractmethod
    def open(self, attrs: dict | None = None) -> None:
        """Open the output file and write global attributes."""

    @abstractmethod
    def write_episode(
        self,
        episode_idx: int,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: dict[str, np.ndarray] | None = None,
        meta: dict | None = None,
    ) -> None:
        """Write one episode's trajectory.

        Args:
            episode_idx: Zero-based episode number.
            observations: Dict mapping key → (T, ...) array.
            actions: (T, A) float32.
            rewards: (T,) float32.
            dones: (T,) bool.
            infos: Optional dict of per-step scalar/vector data.
            meta: Optional episode-level metadata dict.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the output file."""
