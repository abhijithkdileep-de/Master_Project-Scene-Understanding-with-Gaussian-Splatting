"""
example commands for inference:
PYTHONPATH=. python tools/lang_inference.py \
    --config configs/inference/lang-pretrain-pt-v3m1-3dgs.py \
    --checkpoint ckpts/model_best.pth \
    --input-root /path/to/a/preprocessed/3dgs/npy/folder \
    --output-dir /output/path
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from pointcept.inference import LangPretrainerInference
from pointcept.utils.config import Config

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone inference for LangPretrainer."
    )
    parser.add_argument("--config", required=True, help="Path to inference config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file to load.")
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory that stores processed Gaussian .npy files.",
    )
    parser.add_argument("--scene-name", default=None, help="Optional scene name.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory defined in config.save_features.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving even if config enables it.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device string such as cpu or cuda:0.",
    )
    parser.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to save a JSON summary of produced features.",
    )
    parser.add_argument(
        "--pca-vis",
        action="store_true",
        help="Run PCA visualization after inference using the saved feature output.",
    )
    parser.add_argument(
        "--pca-method",
        choices=("pc123_pct01_99_q6", "baseline_mix075_minmax_q6"),
        default="pc123_pct01_99_q6",
        help="PCA color mapping used by --pca-vis.",
    )
    parser.add_argument(
        "--pca-brightness",
        type=float,
        default=1.25,
        help="Brightness multiplier for --pca-vis colors.",
    )
    parser.add_argument(
        "--pca-seed",
        type=int,
        default=1219,
        help="Seed for --pca-vis colorization.",
    )
    return parser.parse_args()


def _default_output_dir():
    return _REPO_ROOT / "outputs"


def _resolve_saved_backbone_path(inferencer, scene_name):
    if not inferencer.save_backbone:
        raise ValueError(
            "`--pca-vis` requires inference.save_features.backbone.enabled=True."
        )
    if not inferencer.output_dir:
        raise ValueError("`--pca-vis` requires an output directory for saved features.")
    file_name = inferencer.backbone_save_cfg.get("file_name", "feat.pt")
    return Path(inferencer._resolve_output_path(scene_name, file_name))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    feat_keys = cfg.get("feat_keys", None)
    if not feat_keys:
        raise KeyError("Inference config must define `feat_keys`.")

    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    data_dict = {}
    for file_name in os.listdir(args.input_root):
        if not file_name.endswith(".npy"):
            continue
        key = os.path.splitext(file_name)[0]
        file_path = os.path.join(args.input_root, file_name)
        data_dict[key] = np.load(file_path)

    missing = [k for k in feat_keys if k not in data_dict]
    if missing:
        raise FileNotFoundError(
            "Missing required feature files: " + ", ".join(f"{m}.npy" for m in missing)
        )

    scene_name = args.scene_name or os.path.basename(os.path.normpath(args.input_root))

    inferencer = LangPretrainerInference(
        cfg,
        args.checkpoint,
        device=args.device,
    )
    if args.output_dir is not None:
        inferencer.output_dir = args.output_dir
    elif args.pca_vis and not inferencer.output_dir:
        inferencer.output_dir = str(_default_output_dir())
    if args.pca_vis and not args.no_save and not inferencer.save_backbone:
        raise ValueError(
            "`--pca-vis` requires inference.save_features.backbone.enabled=True."
        )

    outputs = inferencer(
        data_dict,
        scene_name=scene_name,
        save=not args.no_save,
    )

    backbone = outputs["backbone_features"]
    summary = {
        "name": outputs["name"],
        "backbone_features_shape": (
            list(backbone.shape) if backbone is not None else None
        ),
        "metadata_keys": list(outputs["metadata"].keys()),
    }

    if args.pca_vis:
        if args.no_save:
            if backbone is None:
                raise RuntimeError("PCA visualization requires model features in memory.")
            from scripts.pca_colorize_features import run_pca_visualization_from_features

            pca_summary = run_pca_visualization_from_features(
                features=backbone,
                input_root=args.input_root,
                output_dir=inferencer.output_dir,
                scene_name=outputs["name"],
                pca_method=args.pca_method,
                brightness=args.pca_brightness,
                pca_seed=args.pca_seed,
            )
        else:
            feature_path = _resolve_saved_backbone_path(inferencer, outputs["name"])
            if not feature_path.exists():
                raise FileNotFoundError(
                    f"PCA visualization expected a saved feature file at {feature_path}, "
                    "but it was not found."
                )

            from scripts.pca_colorize_features import run_pca_visualization

            pca_summary = run_pca_visualization(
                feature_path=feature_path,
                input_root=args.input_root,
                output_dir=inferencer.output_dir,
                scene_name=outputs["name"],
                pca_method=args.pca_method,
                brightness=args.pca_brightness,
                pca_seed=args.pca_seed,
            )
        summary["pca_visualization"] = pca_summary

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(summary, f, indent=2)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
