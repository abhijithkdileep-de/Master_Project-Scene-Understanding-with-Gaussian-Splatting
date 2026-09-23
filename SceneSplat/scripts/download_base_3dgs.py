#!/usr/bin/env python3
"""Download base 3DGS releases from Hugging Face without feature files."""

import argparse
import inspect
from pathlib import Path

from huggingface_hub import snapshot_download


REPOS = {
    "scannet": "clapfor/scannet_mcmc_3dgs_lang_base",
    "scannetpp": "clapfor/scannetpp_v2_mcmc_3dgs_lang_base",
    "matterport3d": "clapfor/matterport3d_scene_mcmc_3dgs_lang_base",
}

IGNORE_PATTERNS = [
    "train_grid*/**",
    "val_grid*/**",
    "test_grid*/**",
    "**/valid_feat_mask.npy",
    "**/lang_feat.npy",
    "**/lang_feat_index.npy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the base SceneSplat 3DGS releases and skip pre-chunked "
            "folders plus old language-feature sidecars."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where the downloaded release folders will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(REPOS),
        default=tuple(REPOS),
        help="Datasets to download. Defaults to all datasets.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel Hugging Face download workers.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Optional allow patterns for controlled tests or partial downloads.",
    )
    return parser.parse_args()


def ensure_hf_hub_support() -> None:
    params = inspect.signature(snapshot_download).parameters
    required_params = {
        "repo_type",
        "local_dir",
        "allow_patterns",
        "ignore_patterns",
        "max_workers",
    }
    missing = sorted(required_params.difference(params))
    if missing:
        raise SystemExit(
            "This script needs a newer huggingface_hub with support for "
            f"{', '.join(missing)}. Run it in the SceneSplatPro environment "
            "or install huggingface_hub>=0.14."
        )


def main() -> None:
    args = parse_args()
    ensure_hf_hub_support()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        repo_id = REPOS[dataset]
        local_dir = args.output_root / repo_id.split("/")[-1]
        print(f"Downloading {dataset} from {repo_id} -> {local_dir}")

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=args.allow_patterns,
            ignore_patterns=IGNORE_PATTERNS,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
