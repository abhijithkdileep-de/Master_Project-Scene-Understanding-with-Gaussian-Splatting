"""Export 3DGS parameters from standard or compressed PLY into `.npy` files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gaussian_io import load_gaussian_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .ply file or directory containing .ply files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for exported .npy files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process .ply files in subdirectories.",
    )
    parser.add_argument(
        "--disable-outlier-filter",
        action="store_true",
        help="Disable raw-Ply outlier filtering while exporting.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("chorus.preprocess_gs")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def collect_ply_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".ply":
        return [input_path]
    if input_path.is_dir():
        if recursive:
            return sorted(input_path.rglob("*.ply"))
        return sorted(input_path.glob("*.ply"))
    raise ValueError(f"Input path is not a valid .ply file or directory: {input_path}")


def process_ply_file(
    ply_path: Path,
    output_dir: Path,
    *,
    disable_outlier_filter: bool,
    logger: logging.Logger,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_reader_cfg = dict(outlier_filter=dict(enabled=not disable_outlier_filter))
    try:
        result = load_gaussian_input(
            ply_path,
            required_keys=("color", "opacity", "quat", "scale"),
            input_reader_cfg=input_reader_cfg,
            logger=logger,
        )
    except Exception as exc:
        logger.error("Failed to load %s: %s", ply_path, exc)
        return False

    np.save(output_dir / "coord.npy", result.data["coord"])
    np.save(output_dir / "color.npy", result.data["color"])
    np.save(output_dir / "opacity.npy", result.data["opacity"])
    np.save(output_dir / "scale.npy", result.data["scale"])
    np.save(output_dir / "quat.npy", result.data["quat"])

    logger.info(
        "Exported %s to %s (%s, raw=%d, kept=%d, outlier_filter=%s)",
        ply_path,
        output_dir,
        result.source_type,
        result.raw_count,
        result.kept_count,
        "OFF" if disable_outlier_filter else "ON",
    )
    return True


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    input_path = Path(args.input)
    output_path = Path(args.output)
    ply_files = collect_ply_files(input_path, args.recursive)
    if not ply_files:
        raise RuntimeError(f"No .ply files found under {input_path}")

    logger.info("Found %d PLY file(s) to process.", len(ply_files))
    for ply_file in tqdm(ply_files):
        if input_path.is_dir():
            relative_path = ply_file.relative_to(input_path)
            destination = output_path / relative_path.parent / relative_path.stem
        else:
            destination = output_path
        process_ply_file(
            ply_file,
            destination,
            disable_outlier_filter=args.disable_outlier_filter,
            logger=logger,
        )

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
