"""Robomimic-compatible HDF5 recorder.

Produces datasets loadable directly by robomimic's
dataset = SequenceDataset(hdf5_path=..., ...)

File structure:
    demo.hdf5
    ├── data/
    │   ├── demo_0/
    │   │   ├── states              (T, D)  ← concatenation of state_keys
    │   │   ├── actions             (T, A)
    │   │   ├── rewards             (T,)
    │   │   ├── dones               (T,)
    │   │   ├── obs/                ← all observation keys live here
    │   │   │   ├── <key>           image keys → gzip
    │   │   │   └── ...
    │   │   ├── model_file          MuJoCo XML string
    │   │   └── .attrs/
    │   └── ...
    └── .attrs/
        ├── total
        └── env_args  (JSON)
"""

import json
from pathlib import Path

import h5py
import numpy as np

from domain_rand.pipeline.recorders.base import BaseRecorder


class RobomimicRecorder(BaseRecorder):
    """Writes trajectories in robomimic-compatible HDF5 format.

    Constructor kwargs (passed from il_demo.recorder_kwargs in config):

        image_keys: list[str]
            Observation keys that contain image data (rank ≥ 3).
            These are stored under obs/ with gzip compression.
            Default: ["rgb"].

        state_keys: list[str]
            Observation keys to concatenate into the ``states``
            dataset.  Order determines concatenation order.
            Default: ["state"].

        key_rename: dict[str, str]
            Optional mapping from Task observation keys to output
            dataset names.  e.g. {"rgb": "agentview_image"}.
            Keys not listed here keep their original names.

        store_model_xml: bool
            Whether to store the MuJoCo XML as a string dataset.
            Default: True.
    """

    def __init__(self, output_path: str | Path, **kwargs):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._image_keys: list[str] = kwargs.get("image_keys", ["rgb"])
        self._state_keys: list[str] = kwargs.get("state_keys", ["state"])
        self._key_rename: dict[str, str] = kwargs.get("key_rename", {})
        self._store_model_xml: bool = kwargs.get("store_model_xml", True)

        self._file: h5py.File | None = None
        self._scene_xml: str = ""

    # ── Public API ──────────────────────────────────────────────────

    def set_scene_xml(self, xml_text: str) -> None:
        """Provide the MuJoCo XML string for model_file storage."""
        self._scene_xml = xml_text

    # ── BaseRecorder interface ──────────────────────────────────────

    def open(self, attrs: dict | None = None) -> None:
        self._file = h5py.File(self.output_path, "w")
        data_grp = self._file.create_group("data")

        own_attrs = {
            "total": 0,
            "env_args": json.dumps(attrs or {}, indent=2, ensure_ascii=False),
        }
        for key, value in own_attrs.items():
            self._file.attrs[key] = value

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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

        grp_name = f"demo_{episode_idx}"
        data_grp = self._file["data"]
        if grp_name in data_grp:
            del data_grp[grp_name]
        grp = data_grp.create_group(grp_name)

        # ── Observations ─────────────────────────────────────────
        obs_grp = grp.create_group("obs")

        # Rename keys for output
        renamed = {self._key_rename.get(k, k): v for k, v in observations.items()}

        for out_key, arr in renamed.items():
            is_image = (
                out_key in self._image_keys
                or any(ik in out_key for ik in self._image_keys)
                or arr.ndim >= 3
            )
            if is_image:
                obs_grp.create_dataset(
                    out_key, data=arr,
                    compression="gzip", compression_opts=4,
                )
            else:
                obs_grp.create_dataset(out_key, data=arr)

        # info dict also goes under obs (robomimic convention)
        if infos:
            for ikey, iarr in infos.items():
                if ikey not in obs_grp:
                    obs_grp.create_dataset(ikey, data=iarr)

        # ── States (concatenated) ──────────────────────────────────
        state_parts = []
        for sk in self._state_keys:
            if sk in observations:
                s = observations[sk]
                if s.ndim == 1:
                    s = s.reshape(-1, 1)
                state_parts.append(s)
        if state_parts:
            states = np.concatenate(state_parts, axis=1)
            grp.create_dataset("states", data=states)

        # ── Actions, rewards, dones ────────────────────────────────
        grp.create_dataset("actions", data=actions)
        grp.create_dataset("rewards", data=rewards)
        grp.create_dataset("dones", data=dones)

        # ── Model file ─────────────────────────────────────────────
        if self._store_model_xml and self._scene_xml:
            model_ds = grp.create_dataset("model_file", data=self._scene_xml)
            # Store dtype explicitly as variable-length UTF-8 string
            # h5py auto-handles this for str scalars

        # ── Metadata ───────────────────────────────────────────────
        if meta is not None:
            grp.attrs["meta"] = json.dumps(meta, indent=2, ensure_ascii=False)

        # Update total count
        self._file.attrs["total"] = episode_idx + 1

    @property
    def file(self) -> h5py.File | None:
        return self._file
