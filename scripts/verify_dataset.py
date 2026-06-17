#!/usr/bin/env python3
"""Quick verification of a generated HDF5 dataset."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def verify(dataset_path: str):
    with h5py.File(dataset_path, "r") as f:
        print("=" * 60)
        print("Dataset Verification")
        print("=" * 60)
        print(f"File: {dataset_path}")
        print(f"Top-level attrs: {list(f.attrs.keys())}")
        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, str) and len(val) > 100:
                print(f"  {key}: [truncated, {len(val)} chars]")
            else:
                print(f"  {key}: {val}")

        episodes = sorted([k for k in f.keys() if k.startswith("episode_")])
        print(f"\nEpisodes: {len(episodes)}")

        # Check first episode
        grp = f[episodes[0]]
        print(f"\nFirst episode: {episodes[0]}")
        print(f"  Datasets: {list(grp.keys())}")
        print(f"  Group attrs: {list(grp.attrs.keys())}")

        rgb = grp["rgb"][:]
        print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}, range: [{rgb.min()}, {rgb.max()}]")

        if "depth" in grp:
            depth = grp["depth"][:]
            print(f"  Depth shape: {depth.shape}, dtype: {depth.dtype}, range: [{depth.min():.3f}, {depth.max():.3f}]")

        if "extrinsics" in grp:
            extr = grp["extrinsics"][:]
            print(f"  Extrinsics shape: {extr.shape}")
            print(f"  Extrinsics:\n{extr}")

        if "intrinsics" in grp:
            intr = grp["intrinsics"][:]
            print(f"  Intrinsics shape: {intr.shape}")
            print(f"  Intrinsics:\n{intr}")

        if "meta" in grp.attrs:
            meta = json.loads(grp.attrs["meta"])
            print(f"\n  Meta keys: {list(meta.keys())}")
            print(f"  Camera: {meta.get('camera_name')}")
            print(f"  Camera ID: {meta.get('camera_id')}")
            print(f"  Sim time: {meta.get('sim_time')}")
            print(f"  Episode: {meta.get('episode')}")
            print(f"  Frame: {meta.get('frame')}")

        # Check RGB varies across episodes (confirms randomization works)
        if len(episodes) >= 2:
            rgb0 = f[episodes[0]]["rgb"][:]
            rgb1 = f[episodes[1]]["rgb"][:]
            diff = np.mean(np.abs(rgb0.astype(float) - rgb1.astype(float)))
            print(f"\n  Mean RGB diff between ep0 and ep1: {diff:.1f} (out of 255)")
            if diff > 1.0:
                print("  [OK] Randomization is working (images differ)")
            else:
                print("  [WARN] Images are nearly identical - check randomization config")

        print("\n[OK] Dataset structure verified successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify a generated dataset.")
    parser.add_argument("dataset", type=str, help="Path to HDF5 dataset.")
    args = parser.parse_args()
    verify(args.dataset)
