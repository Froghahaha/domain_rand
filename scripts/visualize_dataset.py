#!/usr/bin/env python3
"""Visualize a generated domain-randomized dataset.

Displays a grid of sampled frames from the HDF5 dataset, showing RGB images
with optional depth overlay.

Usage:
    python scripts/visualize_dataset.py --dataset ./datasets/dataset_xxx.h5
    python scripts/visualize_dataset.py --dataset ./datasets/dataset_xxx.h5 --samples 16
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def visualize_dataset(dataset_path: str, num_samples: int = 16):
    """Load a dataset and display a grid of sample RGB images."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization. Install with: pip install matplotlib")
        sys.exit(1)

    path = Path(dataset_path)
    if not path.exists():
        print(f"Dataset not found: {path}")
        sys.exit(1)

    with h5py.File(path, "r") as f:
        # Read global metadata
        print("=" * 60)
        print("Dataset Info")
        print("=" * 60)
        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, str) and len(val) > 100:
                print(f"  {key}: [truncated, {len(val)} chars]")
            else:
                print(f"  {key}: {val}")

        # List episodes
        episode_keys = sorted([k for k in f.keys() if k.startswith("episode_")])
        num_episodes = len(episode_keys)
        print(f"\nEpisodes: {num_episodes}")

        # Sample episodes
        sample_indices = np.linspace(0, num_episodes - 1, min(num_samples, num_episodes), dtype=int)

        # Determine grid layout
        cols = int(np.ceil(np.sqrt(len(sample_indices))))
        rows = int(np.ceil(len(sample_indices) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.atleast_1d(axes).flatten()

        for ax_idx, ep_idx in enumerate(sample_indices):
            grp_name = episode_keys[ep_idx]
            grp = f[grp_name]

            ax = axes[ax_idx]

            # Read RGB
            if "rgb" in grp:
                rgb = grp["rgb"][:]
                ax.imshow(rgb)
            else:
                ax.text(0.5, 0.5, "No RGB", ha="center", va="center", transform=ax.transAxes)

            # Title with metadata
            if "meta" in grp.attrs:
                meta = json.loads(grp.attrs["meta"])
                cam_name = meta.get("camera_name", "?")
                ax.set_title(f"Ep {ep_idx}\ncam: {cam_name}", fontsize=8)
            else:
                ax.set_title(f"Episode {ep_idx}", fontsize=8)

            ax.axis("off")

        # Hide unused subplots
        for ax_idx in range(len(sample_indices), len(axes)):
            axes[ax_idx].set_visible(False)

        fig.suptitle(f"Domain Randomized Dataset: {path.name}\n({num_episodes} episodes)", fontsize=12)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a domain-randomized dataset.")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Path to HDF5 dataset.")
    parser.add_argument("--samples", "-n", type=int, default=16, help="Number of samples to display.")
    args = parser.parse_args()

    visualize_dataset(args.dataset, args.samples)


if __name__ == "__main__":
    main()
