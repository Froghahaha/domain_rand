"""HDF5 dataset recorder.

Handles creating and writing to an HDF5 file with the following structure:

dataset.h5
├── .attrs/                  # global metadata
├── episode_XXXX/
│   ├── rgb                  # (H, W, 3) uint8
│   ├── depth                # (H, W) float32 (if save_depth)
│   ├── camera_extrinsics    # (4, 4) float32
│   ├── camera_intrinsics    # (3, 3) float32
│   └── .attrs/              # JSON-serialized frame metadata
"""

import json
from pathlib import Path

import h5py
import numpy as np


class DatasetRecorder:
    """Records rendered frames and metadata to an HDF5 file."""

    def __init__(self, output_path: str | Path, mode: str = "w"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._file: h5py.File | None = None

    def open(self, attrs: dict | None = None) -> None:
        """Open the HDF5 file for writing."""
        self._file = h5py.File(self.output_path, self.mode)
        if attrs:
            for key, value in attrs.items():
                self._file.attrs[key] = value

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write_frame(
        self,
        episode_idx: int,
        frame_idx: int,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Write one frame's data to the HDF5 file.

        Args:
            episode_idx: Episode number.
            frame_idx: Frame number within the episode.
            rgb: RGB image (H, W, 3) uint8.
            depth: Optional depth image (H, W) float32.
            extrinsics: Optional 4x4 camera extrinsics matrix.
            intrinsics: Optional 3x3 camera intrinsics matrix.
            metadata: Optional dict of frame metadata.
        """
        if self._file is None:
            raise RuntimeError("Recorder not opened. Call open() first.")

        grp_name = f"episode_{episode_idx:04d}"
        if grp_name not in self._file:
            grp = self._file.create_group(grp_name)
        else:
            grp = self._file[grp_name]

        # Store RGB
        ds_name = f"rgb_{frame_idx:03d}" if frame_idx > 0 else "rgb"
        if ds_name in grp:
            del grp[ds_name]
        grp.create_dataset(ds_name, data=rgb, compression="gzip", compression_opts=4)

        # Store depth
        if depth is not None:
            ds_name = f"depth_{frame_idx:03d}" if frame_idx > 0 else "depth"
            if ds_name in grp:
                del grp[ds_name]
            grp.create_dataset(ds_name, data=depth, compression="gzip", compression_opts=4)

        # Store extrinsics
        if extrinsics is not None:
            ds_name = f"extrinsics_{frame_idx:03d}" if frame_idx > 0 else "extrinsics"
            if ds_name in grp:
                del grp[ds_name]
            grp.create_dataset(ds_name, data=extrinsics)

        # Store intrinsics
        if intrinsics is not None:
            ds_name = f"intrinsics_{frame_idx:03d}" if frame_idx > 0 else "intrinsics"
            if ds_name in grp:
                del grp[ds_name]
            grp.create_dataset(ds_name, data=intrinsics)

        # Store metadata as JSON attribute on the group
        if metadata is not None:
            meta_key = f"meta_{frame_idx:03d}" if frame_idx > 0 else "meta"
            grp.attrs[meta_key] = json.dumps(metadata, indent=2, ensure_ascii=False)

    @property
    def file(self) -> h5py.File | None:
        return self._file
