"""Estimate chunk counts for preprocessed 3DGS scenes from scene bounds."""

import os
import json
import argparse
import numpy as np
from pathlib import Path


def _axis_starts(max_val, stride, size):
    """Generate axis starts for chunking"""
    return np.arange(0, max_val + stride - size, stride)


def count_chunks_for_scene(
    scene_path,
    chunk_range,
    chunk_stride,
    chunk_z,
    max_chunk_num,
):
    """Count chunks for a single scene based on its bounding box"""

    # Read the transforms JSON
    json_path = Path(scene_path) / "transforms_train.json"
    if not json_path.exists():
        print(f"Warning: {json_path} not found, skipping")
        return 0

    with open(json_path, "r") as f:
        data = json.load(f)

    # Get bounding box
    if "bbox_min" not in data or "bbox_max" not in data:
        print(f"Warning: bbox_min/bbox_max not found in {json_path}, skipping")
        return 0

    bbox_min = np.array(data["bbox_min"])
    bbox_max = np.array(data["bbox_max"])

    # Calculate range (assuming we recenter to origin like in original script)
    xyz_range = bbox_max - bbox_min
    max_xyz = xyz_range  # After recentering, min would be ~0, max would be the range

    # Build chunk start grid
    if chunk_z:
        xs = _axis_starts(max_xyz[0], chunk_stride[0], chunk_range[0])
        ys = _axis_starts(max_xyz[1], chunk_stride[1], chunk_range[1])
        zs = _axis_starts(max_xyz[2], chunk_stride[2], chunk_range[2])

        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            num_chunks = 0
        else:
            num_chunks = xs.size * ys.size * zs.size
    else:
        xs = _axis_starts(max_xyz[0], chunk_stride[0], chunk_range[0])
        ys = _axis_starts(max_xyz[1], chunk_stride[1], chunk_range[1])

        if xs.size == 0 or ys.size == 0:
            num_chunks = 0
        else:
            num_chunks = xs.size * ys.size

    # Apply max_chunk_num limit if specified
    if max_chunk_num is not None and num_chunks > max_chunk_num:
        num_chunks = max_chunk_num

    scene_name = Path(scene_path).name
    print(
        f"Scene {scene_name}: {num_chunks} chunks "
        f"(bbox range: [{xyz_range[0]:.2f}, {xyz_range[1]:.2f}, {xyz_range[2]:.2f}])"
    )

    return num_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Estimate number of chunks for 3DGS scenes"
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the dataset root containing scene folders",
    )
    parser.add_argument(
        "--chunk_range",
        default=[6, 6, 6],
        type=float,
        nargs="+",
        help="Range of each chunk, e.g. --chunk_range 6 6 6. With --chunk_z must have 3 values.",
    )
    parser.add_argument(
        "--chunk_stride",
        default=[4, 4, 5],
        type=float,
        nargs="+",
        help="Stride of each chunk, e.g. --chunk_stride 4 4 5. With --chunk_z must have 3 values.",
    )
    parser.add_argument(
        "--max_chunk_num",
        type=int,
        default=None,
        help="Maximum number of chunks to process per scene.",
    )
    parser.add_argument(
        "--chunk_z",
        action="store_true",
        help="Enable 3D chunking along Z axis.",
    )

    args = parser.parse_args()

    # Validate chunk_range/stride lengths
    cr = list(args.chunk_range)
    cs = list(args.chunk_stride)

    if args.chunk_z:
        if len(cr) != 3 or len(cs) != 3:
            raise ValueError(
                "--chunk_z requires --chunk_range and --chunk_stride to each "
                "have exactly 3 values."
            )
    else:
        # Allow 2 or 3 values; ignore Z if provided
        if len(cr) < 2 or len(cs) < 2:
            raise ValueError(
                "--chunk_range/--chunk_stride must have at least 2 values "
                "when --chunk_z is not set."
            )
        if len(cr) > 2:
            print(f"[warn] --chunk_z is off; ignoring extra chunk_range value(s): {cr[2:]}")
            cr = cr[:2]
        if len(cs) > 2:
            print(f"[warn] --chunk_z is off; ignoring extra chunk_stride value(s): {cs[2:]}")
            cs = cs[:2]

    args.chunk_range = tuple(cr)
    args.chunk_stride = tuple(cs)

    # Get all scene directories
    dataset_root = Path(args.dataset_root)
    scene_dirs = [
        d
        for d in dataset_root.iterdir()
        if d.is_dir() and (d / "transforms_train.json").exists()
    ]
    scene_dirs.sort()

    if not scene_dirs:
        print(f"No scene directories found in {dataset_root}")
        return

    print(f"Found {len(scene_dirs)} scenes to process")
    print(f"Chunk configuration:")
    print(f"  Range: {args.chunk_range}")
    print(f"  Stride: {args.chunk_stride}")
    print(f"  3D chunking (Z-axis): {args.chunk_z}")
    print(f"  Max chunks per scene: {args.max_chunk_num}")
    print("=" * 60)

    # Process scenes
    counts = []
    for scene_dir in scene_dirs:
        count = count_chunks_for_scene(
            scene_dir,
            args.chunk_range,
            args.chunk_stride,
            args.chunk_z,
            args.max_chunk_num,
        )
        counts.append(count)

    # Summary
    total_chunks = sum(counts)
    print("=" * 60)
    print(f"[SUMMARY] Total estimated chunks across all scenes: {total_chunks}")
    print(f"  Scenes processed: {len(scene_dirs)}")
    if len(scene_dirs) > 0:
        print(f"  Average chunks per scene: {total_chunks / len(scene_dirs):.2f}")
        print(f"  Min chunks in a scene: {min(counts)}")
        print(f"  Max chunks in a scene: {max(counts)}")


if __name__ == "__main__":
    main()
