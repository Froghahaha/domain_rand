# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the conda environment. Python is at:

```
c:\ProgramData\anaconda3\envs\domain_rand\python.exe
```

Prefix all commands with that path, or use `conda activate domain_rand` if the shell persists state.

## Commands

```bash
# Generate a dataset
python scripts/generate_dataset.py --config configs/default.yaml -n 100 -o ./datasets/test.h5 --seed 42

# Use a DR preset (auto-merged onto defaults — only keys present in preset override)
python scripts/generate_dataset.py --config configs/presets/heavy_visual_dr.yaml -n 500

# Verify HDF5 structure
python scripts/verify_dataset.py datasets/test.h5

# Visualize samples
python scripts/visualize_dataset.py --dataset datasets/test.h5 --samples 16

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_distributions.py -v
```

## Architecture

The pipeline is: **Config → Scene → Randomizers → Render → Record**

### Core abstractions

- **`Scene`** (`core/scene.py`) — wraps `mujoco.MjModel`/`MjData`. Loaded via `MjSpec.from_file()`. Manages nominal state save/restore so randomizers can reset.
- **`Randomizer`** (`randomizers/base.py`) — ABC with three lifecycle methods: `save_nominal(model)`, `randomize(model)`, `restore(model)`. Operates by mutating `mjModel` numpy arrays directly (e.g., `model.geom_rgba[i, :3]`).
- **`DomainRandomizer`** (`randomizers/base.py`) — composite that orchestrates sub-randomizers in registration order. Supports per-randomizer enable/disable flags.
- **`DatasetCollector`** (`pipeline/collector.py`) — main loop: loads scene, builds randomizer from config, then for each episode: randomize → `mj_forward` → render RGB/depth → write HDF5.
- **`DatasetRecorder`** (`pipeline/recorder.py`) — HDF5 writer using `h5py`. Each episode is a group with datasets `rgb`, `depth`, `extrinsics`, `intrinsics`, plus per-frame JSON metadata in `.attrs`.

### Config system

- Dataclass hierarchy: `DomainRandConfig` contains `TextureRandomizationConfig`, `LightingRandomizationConfig`, `CameraRandomizationConfig`, `DatasetConfig`.
- `load_and_merge_config(default_path, preset_path)` — loads both as raw YAML dicts, deep-merges (only keys present in preset override), then converts to dataclass. This prevents default dataclass values (e.g., empty `""` for `scene_path`) in the preset from overwriting valid defaults.
- CLI flags (`--scene`, `--num-episodes`, `--seed`, `--output`) override config values after merge.

### Adding a new randomizer

1. Create a class inheriting `Randomizer` (`randomizers/base.py`) — implement `save_nominal`, `randomize`, `restore`
2. Optionally add `get_state(self, model) -> dict` for recording
3. Add its config dataclass to `core/config.py` and a field on `DomainRandConfig`
4. Register it in `DatasetCollector._build_randomizer()` (`pipeline/collector.py`)

### Rendering

- RGB: `renderer.update_scene(data, camera=cam_id)` then `renderer.render()` → `(H, W, 3) uint8`
- Depth: must call `renderer.enable_depth_rendering()` before `renderer.update_scene()` and `renderer.render()`, then `renderer.disable_depth_rendering()` → `(H, W) float32`
- Camera matrices: use `data.cam_xpos`/`data.cam_xmat` for extrinsics; intrinsics derived from `model.cam_fovy` and renderer dimensions

### MuJoCo version notes (3.9.0)

- `geom_texid` does NOT exist — texture mapping goes through `geom_matid` → `mat_texid`
- `light_directional` does NOT exist — use `light_type` instead
- `<light>` and `<camera>` elements must be inside `<worldbody>` in XML (MjSpec validates this strictly)
