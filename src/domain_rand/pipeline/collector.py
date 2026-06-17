"""Main data collection pipeline.

Orchestrates the full pipeline:
  1. Load scene
  2. For each episode:
     a. Apply randomization
     b. Step simulation if needed
     c. Render RGB + depth
     d. Record frame to HDF5
"""

import logging
import time
from pathlib import Path

import mujoco
import numpy as np

from domain_rand.core.config import (
    DomainRandConfig,
    CameraRandomizationConfig,
    LightingRandomizationConfig,
    TextureRandomizationConfig,
)
from domain_rand.core.scene import Scene
from domain_rand.randomizers.base import DomainRandomizer
from domain_rand.randomizers.texture import TextureRandomizer
from domain_rand.randomizers.lighting import LightingRandomizer
from domain_rand.randomizers.camera import CameraRandomizer
from domain_rand.pipeline.recorder import DatasetRecorder
from domain_rand.pipeline.metadata import (
    build_dataset_attrs,
    build_frame_metadata,
)
from domain_rand.utils.rendering import (
    render_rgb,
    render_depth,
    get_camera_extrinsics,
    get_camera_intrinsics,
)

logger = logging.getLogger(__name__)


class DatasetCollector:
    """Main dataset collection pipeline.

    Usage:
        config = load_config("configs/default.yaml")
        collector = DatasetCollector(config)
        collector.run()
    """

    def __init__(self, config: DomainRandConfig):
        self.config = config

        # Set up RNG for reproducibility
        seed = config.seed if config.seed is not None else int(time.time() * 1e6) % (2**31)
        self.rng = np.random.default_rng(seed)
        logger.info(f"RNG seed: {seed}")

        # Load scene
        self.scene = Scene(config.scene_path)
        logger.info(
            f"Loaded scene: {config.scene_path} "
            f"(geoms={self.scene.ngeom}, lights={self.scene.nlight}, cams={self.scene.ncam})"
        )

        # Create and configure randomizer
        self.randomizer = self._build_randomizer()

        # Save nominal state for all randomizers
        self.scene.save_nominal()
        self.randomizer.save_nominal(self.scene.model)

    def _build_randomizer(self) -> DomainRandomizer:
        """Build the DomainRandomizer from config."""
        dr = DomainRandomizer(self.rng)

        dr.register(
            "texture",
            TextureRandomizer(self.config.texture, self.rng),
            enabled=self.config.texture.enabled,
        )
        dr.register(
            "lighting",
            LightingRandomizer(self.config.lighting, self.rng),
            enabled=self.config.lighting.enabled,
        )
        dr.register(
            "camera",
            CameraRandomizer(self.config.camera, self.rng),
            enabled=self.config.camera.enabled,
        )

        return dr

    def run(self, output_path: str | Path | None = None) -> Path:
        """Run the full dataset collection pipeline.

        Args:
            output_path: Optional explicit output path. If None, auto-generates one.

        Returns:
            Path to the generated HDF5 file.
        """
        cfg = self.config.dataset

        if output_path is not None:
            output_path = Path(output_path)
        else:
            output_path = Path(cfg.output_dir) / f"dataset_{int(time.time())}.h5"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get renderer
        renderer = self.scene.get_renderer(
            height=cfg.render_height, width=cfg.render_width
        )

        attrs = build_dataset_attrs(self.config, cfg.num_episodes)

        with DatasetRecorder(output_path) as recorder:
            recorder.open(attrs)

            for ep in range(cfg.num_episodes):
                # ── Randomize ─────────────────────────────────────────
                self.randomizer.randomize(self.scene.model)

                # Apply changes
                self.scene.forward()

                # Determine active camera
                cam_randomizer = None
                for name, r, _ in self.randomizer._randomizers:
                    if name == "camera":
                        cam_randomizer = r
                        break

                cam_id = cam_randomizer.active_camera if cam_randomizer else 0

                # ── Collect frames ─────────────────────────────────────
                for frame in range(cfg.frames_per_episode):
                    if frame > 0:
                        # Step simulation between frames in multi-frame episodes
                        self.scene.step()

                    t0 = time.perf_counter()

                    # Render RGB
                    rgb = render_rgb(
                        self.scene.model, self.scene.data, renderer, camera=cam_id
                    )

                    # Render depth
                    depth = None
                    if cfg.save_depth:
                        depth = render_depth(
                            self.scene.model, self.scene.data, renderer, camera=cam_id
                        )

                    # Camera matrices
                    extrinsics = get_camera_extrinsics(
                        self.scene.model, self.scene.data, camera=cam_id
                    )
                    intrinsics = get_camera_intrinsics(
                        self.scene.model, renderer, camera=cam_id
                    )

                    # Metadata
                    randomizer_state = self.randomizer.get_state(self.scene.model)
                    t1 = time.perf_counter()
                    meta = build_frame_metadata(
                        episode_idx=ep,
                        frame_idx=frame,
                        model=self.scene.model,
                        data=self.scene.data,
                        camera_id=cam_id,
                        randomizer_state=randomizer_state,
                        timestamp=t1,
                    )

                    # Record
                    recorder.write_frame(
                        episode_idx=ep,
                        frame_idx=frame,
                        rgb=rgb,
                        depth=depth,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        metadata=meta,
                    )

                    render_time = t1 - t0
                    logger.debug(
                        f"Episode {ep:04d}/{cfg.num_episodes}, "
                        f"frame {frame}/{cfg.frames_per_episode} "
                        f"({render_time*1000:.1f}ms)"
                    )

                if (ep + 1) % max(1, cfg.num_episodes // 10) == 0:
                    logger.info(f"Progress: {ep + 1}/{cfg.num_episodes} episodes")

        logger.info(f"Dataset saved to: {output_path}")
        return output_path
