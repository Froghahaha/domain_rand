"""Metadata management for dataset generation.

Tracks per-episode and per-frame metadata: timestamps, randomization
parameters, camera info, etc.
"""

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone

import mujoco
import numpy as np

from domain_rand.core.config import DomainRandConfig


def serialize_array(arr: np.ndarray) -> list:
    """Convert a numpy array to a JSON-serializable list."""
    return arr.tolist()


def serialize_state(state: dict) -> dict:
    """Recursively convert numpy arrays in a state dict to lists."""
    result = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            result[k] = serialize_array(v)
        elif isinstance(v, dict):
            result[k] = serialize_state(v)
        elif isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = float(v)
        else:
            result[k] = v
    return result


def build_dataset_attrs(config: DomainRandConfig, num_episodes: int) -> dict:
    """Build top-level HDF5 attributes from the config.

    Returns a dict of JSON-serializable values for .attrs.
    """
    return {
        "config_json": json.dumps(asdict(config), indent=2, ensure_ascii=False),
        "num_episodes": num_episodes,
        "resolution": [config.dataset.render_height, config.dataset.render_width],
        "save_depth": config.dataset.save_depth,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "creator": "domain_rand",
        "version": "0.1.0",
    }


def build_frame_metadata(
    episode_idx: int,
    frame_idx: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    randomizer_state: dict,
    timestamp: float,
) -> dict:
    """Build per-frame metadata dict.

    Returns a dict suitable for storing as HDF5 attributes or a JSON string.
    """
    meta = {
        "episode": episode_idx,
        "frame": frame_idx,
        "sim_time": float(data.time),
        "camera_id": int(camera_id),
        "camera_name": _get_camera_name(model, camera_id),
        "randomizer_params": serialize_state(randomizer_state),
        "wall_time": timestamp,
    }
    return meta


def _get_camera_name(model: mujoco.MjModel, cam_id: int) -> str:
    """Get camera name from its id."""
    if cam_id < 0 or cam_id >= model.ncam:
        return f"cam_{cam_id}"
    addr = model.name_camadr[cam_id]
    if addr >= 0:
        return model.names[addr:].split(b"\x00")[0].decode()
    return f"cam_{cam_id}"
