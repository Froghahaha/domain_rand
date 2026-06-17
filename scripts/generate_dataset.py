#!/usr/bin/env python3
"""Main entry point for generating a domain-randomized dataset.

Usage:
    python scripts/generate_dataset.py \\
        --config configs/default.yaml \\
        --output ./datasets/my_dataset.h5

    # Override specific settings via the command line:
    python scripts/generate_dataset.py \\
        --config configs/presets/heavy_visual_dr.yaml \\
        --num-episodes 100 \\
        --scene assets/scenes/example_factory.xml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path so we can run this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from domain_rand.core.config import load_config, load_and_merge_config, DomainRandConfig
from domain_rand.pipeline.collector import DatasetCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a domain-randomized dataset from a MuJoCo scene."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--scene", "-s",
        type=str,
        default=None,
        help="Path to MuJoCo XML/MJCF scene file (overrides config).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output HDF5 file path (overrides config).",
    )
    parser.add_argument(
        "--num-episodes", "-n",
        type=int,
        default=None,
        help="Number of episodes to generate (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config: always start from defaults, then overlay user-specified
    default_config_path = Path("configs/default.yaml")
    if not default_config_path.exists():
        logger.error(f"Default config not found: {default_config_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    if config_path.resolve() == default_config_path.resolve():
        config = load_config(default_config_path)
        logger.info(f"Loaded config: {config_path}")
    else:
        config = load_and_merge_config(default_config_path, config_path)
        logger.info(f"Loaded config: {default_config_path} + {config_path}")

    # Apply CLI overrides
    if args.scene:
        config.scene_path = args.scene
    if args.seed is not None:
        config.seed = args.seed
    if args.num_episodes is not None:
        config.dataset.num_episodes = args.num_episodes
    if args.output:
        output_override = args.output
    else:
        output_override = None

    # Validate scene exists
    scene_path = Path(config.scene_path)
    if not scene_path.exists():
        logger.error(f"Scene file not found: {scene_path}")
        sys.exit(1)

    logger.info(f"Scene: {scene_path}")
    logger.info(f"Episodes: {config.dataset.num_episodes}")
    logger.info(f"Resolution: {config.dataset.render_width}x{config.dataset.render_height}")
    logger.info(f"Texture DR: {config.texture.enabled} (mode={config.texture.mode})")
    logger.info(f"Lighting DR: {config.lighting.enabled}")
    logger.info(f"Camera DR: {config.camera.enabled}")
    logger.info(f"Save depth: {config.dataset.save_depth}")
    logger.info(f"Seed: {config.seed}")

    # Create collector and run
    collector = DatasetCollector(config)
    output_path = collector.run(output_path=output_override)

    logger.info(f"Done! Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
