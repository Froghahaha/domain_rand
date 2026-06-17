"""Configuration system for domain randomization.

Uses Python dataclasses for type-safe config, with YAML serialization.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import yaml


# ── Texture randomization config ────────────────────────────────────────────

@dataclass
class TextureRandomizationConfig:
    enabled: bool = True
    # Randomization mode: "rgb" = per-geom color, "checker" = procedural checker
    mode: Literal["rgb", "checker", "noise", "gradient"] = "rgb"
    # RGB range: [R_min, R_max, G_min, G_max, B_min, B_max, A_min, A_max]
    rgba_range: tuple[float, float, float, float, float, float, float, float] = (
        0.2, 1.0,  # R
        0.2, 1.0,  # G
        0.2, 1.0,  # B
        1.0, 1.0,  # A (keep opaque)
    )
    # Per-geom overrides: geom_name -> mode-specific config. Empty = randomize all.
    geom_overrides: dict[str, dict] = field(default_factory=dict)
    # Geoms to exclude from randomization (matched by name substring)
    exclude_geoms: list[str] = field(default_factory=list)


# ── Lighting randomization config ───────────────────────────────────────────

@dataclass
class LightingRandomizationConfig:
    enabled: bool = True
    # Position range for each light (added as offset to nominal position)
    position_jitter: float = 0.5
    # Diffuse color range per channel
    diffuse_range: tuple[float, float] = (0.3, 1.0)
    # Ambient color range per channel
    ambient_range: tuple[float, float] = (0.1, 0.5)
    # Specular color range per channel
    specular_range: tuple[float, float] = (0.0, 0.5)
    # Direction jitter (radians, applied to nominal direction)
    direction_jitter: float = 0.3
    # Randomly toggle lights on/off
    random_toggle: bool = False
    # Probability of a light being on when random_toggle is enabled
    toggle_probability: float = 0.8


# ── Camera randomization config ─────────────────────────────────────────────

@dataclass
class CameraRandomizationConfig:
    enabled: bool = True
    # Position jitter: ± this value around nominal position (per axis)
    position_jitter: tuple[float, float, float] = (0.1, 0.1, 0.05)
    # Rotation jitter: ± this value around nominal quaternion (per component)
    rotation_jitter: float = 0.05
    # Field-of-view range in degrees
    fovy_range: tuple[float, float] = (40.0, 70.0)
    # If multiple cameras, randomly pick one per episode
    random_camera: bool = True


# ── Dataset config ──────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    num_episodes: int = 1000
    frames_per_episode: int = 1
    output_dir: str = "./datasets"
    output_format: Literal["hdf5"] = "hdf5"
    render_height: int = 480
    render_width: int = 640
    save_depth: bool = True
    save_segmentation: bool = False


# ── Top-level config ────────────────────────────────────────────────────────

@dataclass
class DomainRandConfig:
    scene_path: str = ""
    texture: TextureRandomizationConfig = field(default_factory=TextureRandomizationConfig)
    lighting: LightingRandomizationConfig = field(default_factory=LightingRandomizationConfig)
    camera: CameraRandomizationConfig = field(default_factory=CameraRandomizationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seed: int | None = None


# ── YAML I/O ────────────────────────────────────────────────────────────────

def _dict_to_dataclass(d: dict, dataclass_type: type) -> object:
    """Recursively convert a dict to a dataclass instance."""
    field_types = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in d.items():
        if key in field_types:
            expected_type = field_types[key]
            # Handle nested dataclass fields
            if isinstance(value, dict) and hasattr(expected_type, "__dataclass_fields__"):
                kwargs[key] = _dict_to_dataclass(value, expected_type)
            else:
                kwargs[key] = value
    return dataclass_type(**kwargs)


def load_config(path: str | Path) -> DomainRandConfig:
    """Load a DomainRandConfig from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_dataclass(raw, DomainRandConfig)


def save_config(config: DomainRandConfig, path: str | Path) -> None:
    """Save a DomainRandConfig to a YAML file."""
    d = asdict(config)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _load_raw_dict(path: str | Path) -> dict:
    """Load a YAML file as a raw dict (no dataclass conversion)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. override values take precedence.
    Only keys present in override are merged; base keys not in override are kept.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def merge_configs(base: DomainRandConfig, override: DomainRandConfig) -> DomainRandConfig:
    """Merge two configs, with override taking precedence.

    Use load_and_merge_config for file-based merging (avoids default-value overwrite).
    """
    base_dict = asdict(base)
    override_dict = asdict(override)

    # Only include keys from override that differ from its own defaults
    merged = _deep_merge_dict(base_dict, override_dict)
    return _dict_to_dataclass(merged, DomainRandConfig)


def load_and_merge_config(
    base_path: str | Path,
    override_path: str | Path,
) -> DomainRandConfig:
    """Load a base config, then merge in overrides from another file.

    Only keys present in the override file are used — this avoids
    default values (e.g., empty strings) in the override from
    overwriting valid values in the base.
    """
    base_dict = _load_raw_dict(base_path)
    override_dict = _load_raw_dict(override_path)

    merged = _deep_merge_dict(base_dict, override_dict)
    return _dict_to_dataclass(merged, DomainRandConfig)
