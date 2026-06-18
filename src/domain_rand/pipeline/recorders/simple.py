"""Simple flexible-dict HDF5 recorder (default).

Stores each observation key under observations/<key> and each
info key under infos/<key>.  Image-like arrays (rank ≥ 3) are
gzip-compressed automatically.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from domain_rand.pipeline.recorders.base import BaseRecorder


class SimpleRecorder(BaseRecorder):
    """Flexible dict-based HDF5 trajectory recorder.

    File structure:
        dataset.h5
        ├── .attrs/
        └── episode_XXXX/
            ├── observations/
            │   ├── rgb              (T, H, W, 3)  [gzip]
            │   ├── state            (T, D)
            │   └── ...              (any extra keys from Task)
            ├── infos/               [optional]
            │   ├── distance         (T,)
            │   └── ...
            ├── actions              (T, A)
            ├── rewards              (T,)
            ├── dones                (T,)
            └── .attrs/meta          (JSON)
    """

    def __init__(self, output_path: str | Path, **kwargs):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: h5py.File | None = None

    # ── Context manager ─────────────────────────────────────────────

    def open(self, attrs: dict | None = None) -> None:
        self._file = h5py.File(self.output_path, "w")
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

    # ── Write ───────────────────────────────────────────────────────

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
        if self._file is None:
            raise RuntimeError("Recorder not opened. Call open() first.")

        grp_name = f"episode_{episode_idx:04d}"
        if grp_name in self._file:
            del self._file[grp_name]
        grp = self._file.create_group(grp_name)

        # Observations
        obs_grp = grp.create_group("observations")
        for key, arr in observations.items():
            gzip = arr.ndim >= 3
            obs_grp.create_dataset(
                key, data=arr,
                compression="gzip", compression_opts=4,
            ) if gzip else obs_grp.create_dataset(key, data=arr)

        # Info dict
        if infos:
            info_grp = grp.create_group("infos")
            for key, arr in infos.items():
                info_grp.create_dataset(key, data=arr)

        # Actions, rewards, dones
        grp.create_dataset("actions", data=actions)
        grp.create_dataset("rewards", data=rewards)
        grp.create_dataset("dones", data=dones)

        # Episode metadata
        if meta is not None:
            grp.attrs["meta"] = json.dumps(meta, indent=2, ensure_ascii=False)

    @property
    def file(self) -> h5py.File | None:
        return self._file
