#!/usr/bin/env python3
"""Mask unreliable 3D Gaussian segment labels that are far from source points.

This script iterates over processed 3D Gaussian (3DGS) folders, looks up the
corresponding raw point cloud for each scene, and identifies Gaussian centroids
whose nearest neighbour in the source cloud is farther than a user-specified
threshold. The semantic labels for those Gaussians are changed to ``-1`` to mark
them as ignored.

The script is designed for large Matterport3D datasets. KD-trees are cached per
scene to avoid rebuilding them for every chunk, and the script continues past
any recoverable error while logging useful diagnostics. A ``--dry-run`` flag is
available for inspecting the expected edits without touching the label files.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

# Default locations on the hipster-l1 cluster. Override via CLI if needed.
DEFAULT_GS_ROOTS = (
    #### scannet
    # "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs_lang_large/train_grid1.0cm_chunk6x6_stride4x4",
    # "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs_lang_large/val_grid1.0cm_chunk6x6_stride4x4",
    # "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs_lang_large/train",
    # "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs_lang_large/val",

    ### matterport 3d
    "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/"
    "matterport3d_region_mcmc_3dgs_lang_large/test_grid1.0cm_chunk6x6_stride4x4",
    "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_"
    "region_mcmc_3dgs_lang_large/train",
    "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_"
    "region_mcmc_3dgs_lang_large/val",
    "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/"
    "matterport3d_region_mcmc_3dgs_lang_large/val_grid1.0cm_chunk6x6_stride4x4",
)
DEFAULT_ORIGINAL_ROOT = Path(
    "/home/yli7/scratch2/datasets/ptv3_preprocessed/scannet_preprocessed"
)
SPLIT_NAMES = ("train", "val", "test")
MUTABLE_SEGMENT_FILES = ("segment20.npy", "segment200.npy")
MMAP_THRESHOLD_BYTES = 10 * (1 << 30)


@dataclass
class ProcessingStats:
    total_folders: int = 0
    updated_folders: int = 0
    skipped_folders: int = 0
    missing_files: int = 0
    missing_source: int = 0
    invalid_geometry: int = 0
    total_points: int = 0
    masked_points: int = 0

    def merge(self, other: "ProcessingStats") -> None:
        self.total_folders += other.total_folders
        self.updated_folders += other.updated_folders
        self.skipped_folders += other.skipped_folders
        self.missing_files += other.missing_files
        self.missing_source += other.missing_source
        self.invalid_geometry += other.invalid_geometry
        self.total_points += other.total_points
        self.masked_points += other.masked_points


class SceneDataCache:
    """Caches one scene's KD-tree at a time to limit memory use."""

    def __init__(self, original_root: Path, mmap_threshold_bytes: int) -> None:
        self._root = original_root
        self._threshold = mmap_threshold_bytes
        self._current_key: Optional[Tuple[str, str]] = None
        self._coords: Optional[np.ndarray] = None
        self._kdtree: Optional[cKDTree] = None

    def query(
        self, split: str, scene: str, points: np.ndarray, workers: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._current_key != (split, scene):
            self._load_scene(split, scene)
        if self._kdtree is None:
            raise RuntimeError("KD-tree not initialised")
        distances, indices = self._kdtree.query(points, k=1, workers=workers)
        return distances, indices

    def _load_scene(self, split: str, scene: str) -> None:
        scene_dir = self._root / split / scene
        coord_path = scene_dir / "coord.npy"
        if not coord_path.exists():
            raise FileNotFoundError(coord_path)
        coords = self._load_coords(coord_path)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Unexpected coord shape for {coord_path}: {coords.shape}")
        if coords.dtype != np.float32:
            coords = np.asarray(coords, dtype=np.float32)
        self._coords = coords
        self._kdtree = cKDTree(coords)
        self._current_key = (split, scene)

    def _load_coords(self, path: Path) -> np.ndarray:
        size_bytes = path.stat().st_size
        if size_bytes > self._threshold:
            return np.load(path, mmap_mode="r")
        return np.load(path)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mask 3D Gaussian segment labels whose nearest point in the source "
            "point cloud is farther than a distance threshold."
        )
    )
    parser.add_argument(
        "--gs-roots",
        metavar="DIR",
        nargs="+",
        default=list(DEFAULT_GS_ROOTS),
        help="Gaussian scene roots to process (default: configured Matterport3D paths).",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=DEFAULT_ORIGINAL_ROOT,
        help="Root directory of the original point clouds organised by split/scene.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Distance threshold in metres for invalid labels (default: 0.35m).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker threads for KD-tree queries (-1 uses all cores).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying any files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--output_log",
        "--output-log",
        dest="output_log",
        nargs="?",
        const="__DEFAULT__",
        help=(
            "Optional CSV path for per-folder statistics. "
            "If omitted, no file is written. If provided without a value, "
            "the file is created alongside this script."
        ),
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def infer_split(gs_root: Path) -> str:
    name = gs_root.name.lower()
    for split in SPLIT_NAMES:
        if name == split or name.startswith(f"{split}_"):
            return split
    raise ValueError(f"Cannot infer split from directory name: {gs_root}")


def resolve_scene_id(original_root: Path, split: str, folder_name: str) -> Tuple[str, Path]:
    """Return the scene id and directory for a processed folder."""
    candidate = folder_name
    scene_dir = original_root / split / candidate
    if scene_dir.is_dir():
        return candidate, scene_dir
    if "_" in folder_name:
        candidate = folder_name.rsplit("_", 1)[0]
        scene_dir = original_root / split / candidate
        if scene_dir.is_dir():
            return candidate, scene_dir
    raise FileNotFoundError(
        f"Original scene folder not found for processed directory '{folder_name}'"
    )


def load_gaussian_coords(path: Path) -> np.ndarray:
    coords = np.load(path)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Invalid 3DGS coord shape at {path}: {coords.shape}")
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)
    return coords


def check_segment_shapes(coord_count: int, segment_paths: Iterable[Path]) -> bool:
    ok = True
    for seg_path in segment_paths:
        try:
            seg = np.load(seg_path, mmap_mode="r")
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to read %s: %s", seg_path, exc)
            ok = False
            continue
        if seg.ndim != 1 or seg.shape[0] != coord_count:
            logging.error(
                "Segment array at %s has shape %s, expected (%d,)",
                seg_path,
                seg.shape,
                coord_count,
            )
            ok = False
        del seg
    return ok


def mask_segments(segment_paths: Iterable[Path], mask: np.ndarray) -> None:
    for seg_path in segment_paths:
        seg = np.load(seg_path, mmap_mode="r+")
        try:
            seg[mask] = -1
            seg.flush()
        finally:
            del seg


def process_gaussian_folder(
    folder: Path,
    split: str,
    cache: SceneDataCache,
    args: argparse.Namespace,
    root_name: str,
    log_records: Optional[List[Tuple[str, str, int, int, float, float]]] = None,
) -> ProcessingStats:
    stats = ProcessingStats(total_folders=1)
    coord_path = folder / "coord.npy"
    segment_paths = [folder / name for name in MUTABLE_SEGMENT_FILES]

    missing = [p for p in [coord_path, *segment_paths] if not p.exists()]
    if missing:
        logging.error("Missing required files in %s: %s", folder, ", ".join(map(str, missing)))
        stats.missing_files += 1
        stats.skipped_folders += 1
        return stats

    try:
        scene_id, _ = resolve_scene_id(args.original_root, split, folder.name)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        stats.missing_source += 1
        stats.skipped_folders += 1
        return stats

    try:
        coords = load_gaussian_coords(coord_path)
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to load coords at %s: %s", coord_path, exc)
        stats.invalid_geometry += 1
        stats.skipped_folders += 1
        return stats

    if not check_segment_shapes(coords.shape[0], segment_paths):
        stats.invalid_geometry += 1
        stats.skipped_folders += 1
        return stats

    stats.total_points += coords.shape[0]

    try:
        distances, _ = cache.query(split, scene_id, coords, args.workers)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        stats.missing_source += 1
        stats.skipped_folders += 1
        return stats
    except Exception as exc:  # noqa: BLE001
        logging.error("KD-tree query failed for scene %s: %s", scene_id, exc)
        stats.skipped_folders += 1
        return stats

    invalid_mask = distances > args.threshold
    invalid_count = int(np.count_nonzero(invalid_mask))
    stats.masked_points += invalid_count

    total_points = int(coords.shape[0])
    inlier_ratio = float(1.0 - (invalid_count / total_points)) if total_points else 0.0

    if log_records is not None:
        log_records.append(
            (
                root_name,
                folder.name,
                total_points,
                invalid_count,
                float(args.threshold),
                inlier_ratio,
            )
        )

    if invalid_count == 0:
        logging.debug("%s: all %d Gaussians within threshold.", folder, coords.shape[0])
        return stats

    logging.info(
        "%s: %d / %d Gaussians exceed %.3fm",
        folder,
        invalid_count,
        coords.shape[0],
        args.threshold,
    )

    if args.dry_run:
        return stats

    try:
        mask_segments(segment_paths, invalid_mask)
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to write segments in %s: %s", folder, exc)
        stats.skipped_folders += 1
        return stats

    stats.updated_folders += 1
    return stats


def iter_gaussian_folders(root: Path) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry


def process_root(
    gs_root: Path,
    cache: SceneDataCache,
    args: argparse.Namespace,
    log_records: Optional[List[Tuple[str, str, int, int, float, float]]] = None,
) -> ProcessingStats:
    split = infer_split(gs_root)
    logging.info("Processing %s (split=%s)", gs_root, split)
    root_stats = ProcessingStats()
    if not gs_root.exists():
        logging.error("Gaussian root %s does not exist", gs_root)
        root_stats.skipped_folders += 1
        return root_stats

    root_name = gs_root.name

    for folder in iter_gaussian_folders(gs_root):
        folder_stats = process_gaussian_folder(
            folder,
            split,
            cache,
            args,
            root_name,
            log_records,
        )
        root_stats.merge(folder_stats)
    logging.info(
        "%s summary: %d folders, %d updated, %d masked points (dry-run=%s)",
        gs_root,
        root_stats.total_folders,
        root_stats.updated_folders,
        root_stats.masked_points,
        args.dry_run,
    )
    return root_stats


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    log_records: List[Tuple[str, str, int, int, float, float]] = []

    output_log_path: Optional[Path] = None
    if args.output_log:
        if args.output_log == "__DEFAULT__":
            output_log_path = Path(__file__).resolve().with_name(
                "invalidate_far_segments_log.csv"
            )
        else:
            output_log_path = Path(args.output_log)

    original_root = args.original_root
    if not original_root.exists():
        logging.error("Original root %s does not exist", original_root)
        return 1

    cache = SceneDataCache(original_root, MMAP_THRESHOLD_BYTES)
    overall = ProcessingStats()

    for root_str in args.gs_roots:
        gs_root = Path(root_str)
        try:
            root_stats = process_root(
                gs_root,
                cache,
                args,
                log_records if output_log_path else None,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("Unexpected error while processing %s: %s", gs_root, exc)
            continue
        overall.merge(root_stats)

    if output_log_path is not None:
        try:
            output_log_path.parent.mkdir(parents=True, exist_ok=True)
            with output_log_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "root_folder",
                        "subfolder",
                        "total_points",
                        "invalid_points",
                        "threshold",
                        "inlier_ratio",
                    ]
                )
                writer.writerows(log_records)
            logging.info("Wrote CSV log to %s", output_log_path)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to write CSV log to %s: %s", output_log_path, exc)

    logging.info(
        "Overall: %d folders visited, %d updated, %d masked points, %d skipped (dry-run=%s)",
        overall.total_folders,
        overall.updated_folders,
        overall.masked_points,
        overall.skipped_folders,
        args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
