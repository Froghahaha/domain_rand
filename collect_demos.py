#!/usr/bin/env python3
"""Interactive demo collection for imitation learning.

Usage:
    # Collect demos with a custom task and keyboard teleop
    python scripts/collect_demos.py \\
        --config configs/demo_base.yaml \\
        --task my_tasks.push_t.PushTTask \\
        --policy domain_rand.policy.keyboard_teleop.KeyboardTeleop \\
        -n 50 -o ./datasets/push_t_demos.h5

    # Use a scene-level config (visual DR settings are taken from here)
    python scripts/collect_demos.py \\
        --config configs/table_stl.yaml \\
        --task my_tasks.inspect.InspectionTask \\
        --policy domain_rand.policy.keyboard_teleop.KeyboardTeleop
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

# Add src + project root to path so scripts and local task modules can be found
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from domain_rand.core.config import load_config, load_and_merge_config
from domain_rand.pipeline.demo_collector import DemoCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_demos")


def _import_class(dotted_path: str):
    """Import a class from 'module.path.ClassName' string."""
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Expected 'module.ClassName', got '{dotted_path}'"
        )
    mod = importlib.import_module(parts[0])
    return getattr(mod, parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Collect imitation-learning demos with domain randomization."
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/demo_base.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--task", "-t", type=str, required=True,
        help="Fully-qualified Task class name (e.g. 'my_tasks.push_t.PushTTask').",
    )
    parser.add_argument(
        "--policy", "-p", type=str,
        default="domain_rand.policy.keyboard_teleop.KeyboardTeleop",
        help="Fully-qualified Policy class name.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--num-demos", "-n", type=int, default=None,
        help="Number of demo episodes to collect.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--recorder", "-r", type=str, default=None,
        help="Recorder backend: 'simple' (default), 'robomimic', or custom class path.",
    )
    parser.add_argument(
        "--recorder-kwargs", type=str, default=None,
        help="JSON string of extra kwargs for the recorder (e.g. '{\"image_keys\":[\"rgb\"]}').",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load config ──────────────────────────────────────────────────
    default_config_path = Path("configs/default.yaml")
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    if default_config_path.exists() and config_path.resolve() != default_config_path.resolve():
        config = load_and_merge_config(default_config_path, config_path)
        logger.info(f"Loaded config: {default_config_path} + {config_path}")
    else:
        config = load_config(config_path)
        logger.info(f"Loaded config: {config_path}")

    if args.seed is not None:
        config.seed = args.seed
    if args.num_demos is not None:
        config.il_demo.num_demos = args.num_demos
    if args.recorder is not None:
        config.il_demo.recorder = args.recorder
    if args.recorder_kwargs is not None:
        import json
        config.il_demo.recorder_kwargs = json.loads(args.recorder_kwargs)

    # ── Instantiate Task and Policy ──────────────────────────────────
    logger.info(f"Task: {args.task}")
    TaskClass = _import_class(args.task)
    task = TaskClass()

    logger.info(f"Policy: {args.policy}")
    PolicyClass = _import_class(args.policy)
    policy = PolicyClass()

    # ── Run ──────────────────────────────────────────────────────────
    collector = DemoCollector(config, task, policy)
    output_path = collector.run(output_path=args.output)

    logger.info(f"Done! Demos saved to: {output_path}")


if __name__ == "__main__":
    main()
