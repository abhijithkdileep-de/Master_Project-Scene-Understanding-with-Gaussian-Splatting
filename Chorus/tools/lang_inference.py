from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ALIASES = {
    "chorus_3dgs": _REPO_ROOT / "configs/inference/lang-enc-pretrain-chorus-3dgs.py",
    "chorus_pts": _REPO_ROOT / "configs/inference/lang-enc-pretrain-chorus-from-pts-params.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="python -m tools.lang_inference",
        description="Run standalone Chorus inference on a preprocessed scene folder or raw 3DGS PLY."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an inference config, or alias: chorus_3dgs, chorus_pts.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local checkpoint path or Hugging Face checkpoint name/stem.",
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Scene directory with .npy files, or a raw/compressed .ply file.",
    )
    parser.add_argument("--scene-name", default=None, help="Optional scene name.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory. Defaults to config.save_features.output_dir or <repo_root>/outputs.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving features even if enabled in config.",
    )
    parser.add_argument(
        "--disable-outlier-filter",
        action="store_true",
        help="Disable raw-Ply outlier filtering for this run.",
    )
    parser.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to dump a JSON summary of produced features.",
    )
    parser.add_argument(
        "--pca_vis",
        action="store_true",
        help="Run PCA visualization after inference using the saved feature output.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("chorus.inference")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def _json_text(value) -> str:
    return json.dumps(value, indent=2, sort_keys=False)


def _resolve_config(config_value: str) -> dict:
    alias_path = _CONFIG_ALIASES.get(config_value)
    if alias_path is not None:
        return dict(requested=config_value, resolved=str(alias_path), alias=config_value)
    resolved_path = Path(config_value).expanduser().resolve()
    return dict(requested=config_value, resolved=str(resolved_path), alias=None)


def _resolve_output_dir(args_output_dir: str | None, cfg) -> Path:
    from scripts.gaussian_io import get_default_output_dir

    if args_output_dir:
        return Path(args_output_dir).expanduser()
    save_cfg = cfg.inference.get("save_features", {}) or {}
    configured_output_dir = save_cfg.get("output_dir")
    if configured_output_dir:
        return Path(configured_output_dir).expanduser()
    return get_default_output_dir()


def _override_input_reader_cfg(cfg, disable_outlier_filter: bool):
    input_reader_cfg = copy.deepcopy(cfg.inference.get("input_reader", {}))
    if disable_outlier_filter:
        if "outlier_filter" not in input_reader_cfg:
            input_reader_cfg["outlier_filter"] = {}
        input_reader_cfg["outlier_filter"]["enabled"] = False
    return input_reader_cfg


def _select_pca_target(inferencer) -> dict | None:
    if "lang" in inferencer.active_teachers:
        return dict(kind="teacher", name="lang")
    if inferencer.active_teachers:
        return dict(kind="teacher", name=inferencer.active_teachers[0])
    if inferencer.save_backbone:
        return dict(kind="backbone", name="backbone")
    return None


def _resolve_pca_feature_path(
    inferencer, scene_name: str, target: dict
) -> Path:
    if target["kind"] == "teacher":
        teacher_name = target["name"]
        file_name = inferencer.teacher_save[teacher_name].get("file_name", "feat.pt")
        return Path(
            inferencer._resolve_output_path(scene_name, file_name, teacher=teacher_name)
        )
    return Path(inferencer._resolve_output_path(scene_name, "feat.pt", teacher="backbone"))


def _log_run_summary(
    logger,
    read_result,
    runtime_summary,
    config_summary,
    save_enabled,
    pca_target,
):
    checkpoint = runtime_summary["checkpoint"]
    outlier_status = "ON" if read_result.outlier_filter_enabled else "OFF"
    logger.info(
        "Inference setup:\n"
        "  scene_name=%s\n"
        "  input_path=%s\n"
        "  input_type=%s\n"
        "  config_request=%s\n"
        "  config_resolved=%s\n"
        "  checkpoint_source=%s\n"
        "  checkpoint_request=%s\n"
        "  checkpoint_resolved=%s\n"
        "  raw_splats=%d\n"
        "  kept_splats=%d\n"
        "  outlier_filter=%s\n"
        "  save_enabled=%s\n"
        "  output_dir=%s\n"
        "  pca_vis=%s\n"
        "  pca_target=%s\n"
        "  chunk_size=%s",
        read_result.scene_name,
        read_result.input_path,
        read_result.source_type,
        config_summary["requested"],
        config_summary["resolved"],
        checkpoint["source"],
        checkpoint["requested"],
        checkpoint["resolved_name"],
        read_result.raw_count,
        read_result.kept_count,
        outlier_status,
        save_enabled,
        runtime_summary["output_dir"],
        pca_target is not None,
        pca_target["name"] if pca_target is not None else None,
        runtime_summary["chunk_size"],
    )
    if read_result.outlier_filter_enabled and read_result.filter_config is not None:
        logger.info("Active outlier filter config:\n%s", _json_text(read_result.filter_config))
    logger.info("test_cfg:\n%s", _json_text(runtime_summary["test_cfg"]))
    logger.info("save_features:\n%s", _json_text(runtime_summary["save_features"]))


def _log_output_summary(logger, summary):
    logger.info(
        "Inference outputs:\n"
        "  scene_name=%s\n"
        "  teacher_features=%s\n"
        "  backbone_features_shape=%s",
        summary["name"],
        summary["teacher_features"],
        summary["backbone_features_shape"],
    )


def main():
    args = parse_args()
    logger = setup_logger()

    if args.pca_vis and args.no_save:
        raise ValueError("`--pca_vis` requires saving features; remove `--no-save`.")

    from pointcept.inference import LangPretrainerInference
    from pointcept.utils.config import Config
    from scripts.gaussian_io import load_gaussian_input

    config_summary = _resolve_config(args.config)
    cfg = Config.fromfile(config_summary["resolved"])
    feat_keys = cfg.get("feat_keys", None)
    if feat_keys is None:
        raise KeyError("`feat_keys` must be defined in the inference config.")

    inferencer = LangPretrainerInference(cfg, args.checkpoint)
    inferencer.output_dir = str(_resolve_output_dir(args.output_dir, cfg))

    pca_target = None
    if args.pca_vis:
        pca_target = _select_pca_target(inferencer)
        if pca_target is None:
            raise ValueError(
                "`--pca_vis` requires at least one enabled teacher save target or backbone save target."
            )

    output_dir_path = Path(inferencer.output_dir)
    if not args.no_save or args.pca_vis:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    input_reader_cfg = _override_input_reader_cfg(cfg, args.disable_outlier_filter)
    read_result = load_gaussian_input(
        args.input_root,
        feat_keys,
        scene_name=args.scene_name,
        input_reader_cfg=input_reader_cfg,
        logger=logger,
    )

    runtime_summary = inferencer.describe_runtime()

    _log_run_summary(
        logger,
        read_result,
        runtime_summary,
        config_summary,
        save_enabled=not args.no_save,
        pca_target=pca_target,
    )

    outputs = inferencer(
        read_result.data,
        scene_name=read_result.scene_name,
        save=not args.no_save,
        metadata=dict(
            source_input_path=read_result.input_path,
            source_type=read_result.source_type,
            source_raw_count=read_result.raw_count,
            source_kept_count=read_result.kept_count,
            source_keep_index=read_result.kept_indices,
            outlier_filter_enabled=read_result.outlier_filter_enabled,
            outlier_filter_report=read_result.filter_report,
        ),
    )

    summary = {
        "name": outputs["name"],
        "config": config_summary,
        "input": read_result.to_summary(),
        "runtime": runtime_summary,
        "teacher_features": {
            key: list(value.shape) for key, value in outputs["teacher_features"].items()
        },
        "backbone_features_shape": (
            list(outputs["backbone_features"].shape)
            if outputs["backbone_features"] is not None
            else None
        ),
        "metadata_keys": sorted(outputs["metadata"].keys()),
    }

    _log_output_summary(logger, summary)

    if args.pca_vis:
        feature_path = _resolve_pca_feature_path(inferencer, outputs["name"], pca_target)
        if not feature_path.exists():
            raise FileNotFoundError(
                f"PCA visualization expected a saved feature file at {feature_path}, but it was not found."
            )
        from scripts.pca_colorize_features import run_pca_visualization

        pca_summary = run_pca_visualization(
            feature_path=feature_path,
            input_root=args.input_root,
            output_dir=output_dir_path,
            scene_name=outputs["name"],
            device="cuda",
            logger=logger,
        )
        summary["pca_visualization"] = pca_summary
        logger.info(
            "PCA outputs:\n"
            "  feature_path=%s\n"
            "  point_cloud=%s\n"
            "  featvis_3dgs=%s",
            pca_summary["feature_path"],
            pca_summary["point_cloud_path"],
            pca_summary["featvis_path"],
        )

    if args.dump_json:
        output_path = Path(args.dump_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Wrote JSON summary to %s", output_path)


if __name__ == "__main__":
    main()
