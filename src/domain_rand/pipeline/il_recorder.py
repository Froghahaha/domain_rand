"""Imitation-learning-compatible HDF5 trajectory recorder.

Stores multi-frame episodes as time-series arrays, following a format
compatible with robomimic / common IL training pipelines.

File structure:
    dataset.h5
    ├── .attrs/                  # global metadata
    └── episode_XXXX/
        ├── observations/
        │   ├── rgb              # (T, H, W, 3) uint8  [gzip]
        │   ├── state            # (T, D) float32
        │   └── depth            # (T, H, W) float32  [optional]
        ├── actions              # (T, A) float32
        ├── rewards              # (T,) float32
        ├── dones                # (T,) bool
        └── .attrs/              # episode-level metadata (JSON)
"""

import json
from pathlib import Path

import h5py
import numpy as np


class ILRecorder:
    """Records multi-frame IL trajectories to HDF5.

    Usage:
        recorder = ILRecorder("demos.h5")
        recorder.open(global_attrs)
        ...
        recorder.write_episode(ep_idx, rgb, state, action, reward, done, meta)
        recorder.close()
    """

    def __init__(self, output_path: str | Path, mode: str = "w"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._file: h5py.File | None = None

    # ── Context manager ──────────────────────────────────────────────────

    def open(self, attrs: dict | None = None) -> None:
        """Open the HDF5 file and write global attributes."""
        self._file = h5py.File(self.output_path, self.mode)
        if attrs:
            for key, value in attrs.items():
                self._file.attrs[key] = value

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Write ────────────────────────────────────────────────────────────

    def write_episode(
        self,
        episode_idx: int,
        rgb_stack: np.ndarray,
        state_stack: np.ndarray,
        action_stack: np.ndarray,
        reward_stack: np.ndarray,
        done_stack: np.ndarray,
        depth_stack: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> None:
        """Write one episode's trajectory.

        Args:
            episode_idx: Zero-based episode number.
            rgb_stack:  (T, H, W, 3) uint8 images.
            state_stack: (T, D) float32 state vectors.
            action_stack: (T, A) float32 action vectors.
            reward_stack: (T,) float32 rewards.
            done_stack: (T,) bool termination flags.
            depth_stack: Optional (T, H, W) float32 depth images.
            meta: Optional episode-level metadata dict.
        """
        if self._file is None:
            raise RuntimeError("ILRecorder not opened. Call open() first.")

        grp_name = f"episode_{episode_idx:04d}"
        if grp_name in self._file:
            del self._file[grp_name]
        grp = self._file.create_group(grp_name)

        obs_grp = grp.create_group("observations")
        obs_grp.create_dataset(
            "rgb", data=rgb_stack,
            compression="gzip", compression_opts=4,
        )
        obs_grp.create_dataset(
            "state", data=state_stack,
        )

        if depth_stack is not None:
            obs_grp.create_dataset(
                "depth", data=depth_stack,
                compression="gzip", compression_opts=4,
            )

        grp.create_dataset("actions", data=action_stack)
        grp.create_dataset("rewards", data=reward_stack)
        grp.create_dataset("dones", data=done_stack)

        if meta is not None:
            grp.attrs["meta"] = json.dumps(meta, indent=2, ensure_ascii=False)

    # ── Convenience ──────────────────────────────────────────────────────

    @property
    def file(self) -> h5py.File | None:
        return self._file
