import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from pointcept.inference import LangPretrainerInference
from pointcept.utils.config import Config
from pointcept.utils.instance_seg_evaluator import InstanceSegmentationEvaluator
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    HEAD_CLASSES_200,
    COMMON_CLASSES_200,
    TAIL_CLASSES_200,
)

SCANNET200_SUBSETS = {
    "head": HEAD_CLASSES_200,
    "common": COMMON_CLASSES_200,
    "tail": TAIL_CLASSES_200,
}

"""
PYTHONPATH=. python /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/tools/ov_insseg_eval.py
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-vocabulary instance segmentation evaluation with Mask3D proposals."
    )
    parser.add_argument(
        "--config",
        default="configs/inference/lang-enc-pretrain-multi-teacher-from-pts-params.py",
        help="Inference config for LangPretrainerMultiTeacher.",
    )
    parser.add_argument(
        "--checkpoint",
        default="exp_runs/lang_pretrainer/multi_teacher/lang-enc-pretrain-multi-teacher-sn-mcmc-from-pts-params-new/model/best_lang.pth",
        help="Checkpoint to load.",
    )
    parser.add_argument(
        "--data-root",
        default="/gpfs/work3/0/prjs1291/datasets/ptv3_preprocessed/scannet_preprocessed/val",
        help="Root directory containing processed ScanNet scenes.",
    )
    parser.add_argument(
        "--mask-root",
        default="/gpfs/work3/0/prjs1291/datasets/mosaic3d/data/scannet200_masks",
        help="Directory that stores Mask3D proposals (.npz per scene).",
    )
    parser.add_argument(
        "--text-embeddings",
        default="pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2_so400m.pt",
        help="Precomputed SigLIP2 text embeddings (.pt).",
    )
    parser.add_argument(
        "--labels-file",
        default="pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt",
        help="Text file listing class names (one per line).",
    )
    parser.add_argument(
        "--scenes-list",
        default=None,
        help="Optional file with scene ids to evaluate (one per line). "
        "If omitted, all sub-directories under data-root are used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of scenes for quick sanity checks.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string for LangPretrainer inference (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--eval-device",
        default=None,
        help="Device string for similarity + mask aggregation. "
        "Defaults to the inference device.",
    )
    parser.add_argument(
        "--teacher-name",
        default="lang",
        help="Teacher projector name to use for open-vocab evaluation.",
    )
    parser.add_argument(
        "--min-region-size",
        type=int,
        default=100,
        help="Minimum points per mask for evaluator (matches Mosaic3D).",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to dump the final metrics as JSON.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often (scenes) to log progress.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ov_insseg_eval")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def read_class_labels(path: Union[str, os.PathLike[str]]) -> List[str]:
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


def build_subset_mapper(labels: Sequence[str]) -> Dict[str, str]:
    mapper: Dict[str, str] = {"subset_names": ["head", "common", "tail"]}

    def normalize(name: str) -> str:
        return name.lower().replace(" ", "").replace("-", "")

    normalized_map: Dict[str, str] = {}
    for subset, names in SCANNET200_SUBSETS.items():
        for item in names:
            normalized_map[normalize(item)] = subset

    for label in labels:
        norm = normalize(label)
        mapper[label] = normalized_map.get(norm, "tail")
    return mapper


def compute_instance_ignore_idx(labels: Sequence[str]) -> List[int]:
    ignore_idx = []
    for idx, label in enumerate(labels):
        low = label.lower()
        if low in {"wall", "floor"} or "other" in low:
            ignore_idx.append(idx)
    return ignore_idx


def discover_scenes(
    data_root: Path, list_file: Union[str, None], limit: Union[int, None]
) -> List[str]:
    if list_file:
        with open(list_file, "r") as f:
            scenes = [line.strip() for line in f.readlines() if line.strip()]
    else:
        scenes = sorted(
            [
                name
                for name in os.listdir(data_root)
                if (data_root / name).is_dir() and not name.startswith(".")
            ]
        )

    if limit is not None:
        scenes = scenes[:limit]
    return scenes


def load_scene_arrays(
    scene_dir: Path, feat_keys: Sequence[str]
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    scene_data: Dict[str, np.ndarray] = {}
    required = set(feat_keys) | {"coord"}
    for key in required:
        npy_path = scene_dir / f"{key}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Missing required file for key '{key}': {npy_path}"
            )
        scene_data[key] = np.load(npy_path)

    segment_path = scene_dir / "segment200.npy"
    if not segment_path.exists():
        raise FileNotFoundError(f"Missing segment file: {segment_path}")
    segment = np.load(segment_path).astype(np.int32)
    scene_data["segment"] = segment.copy()

    instance_path = scene_dir / "instance.npy"
    if not instance_path.exists():
        raise FileNotFoundError(f"Missing instance file: {instance_path}")
    instance = np.load(instance_path).astype(np.int32)

    if segment.shape[0] != instance.shape[0]:
        raise ValueError(
            f"segment ({segment.shape[0]}) and instance ({instance.shape[0]}) mismatch in {scene_dir}"
        )
    return scene_data, segment, instance


def load_mask_binary(mask_root: Path, scene_name: str) -> np.ndarray:
    mask_path = mask_root / f"{scene_name}.npz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask proposal file: {mask_path}")
    with np.load(mask_path) as data:
        if "masks_binary" not in data:
            raise KeyError(f"'masks_binary' not found in {mask_path}")
        masks = data["masks_binary"].astype(bool)
    return masks


def aggregate_masks(
    logits: torch.Tensor,
    masks_binary: torch.Tensor,
    ignore_class_idx: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if masks_binary.ndim != 2 or masks_binary.shape[1] != logits.shape[0]:
        raise ValueError("Mask tensor shape does not match number of points.")

    mask_list = []
    logits_per_mask = []
    pred_logits = logits.clone()
    if ignore_class_idx:
        neg_inf = torch.finfo(pred_logits.dtype).min
        pred_logits[:, list(ignore_class_idx)] = neg_inf
    pred_logits = F.softmax(pred_logits, dim=-1)

    for mask in masks_binary:
        if mask.sum() == 0:
            continue
        logits_per_mask.append(pred_logits[mask].mean(dim=0))
        mask_list.append(mask)

    if not logits_per_mask:
        return (
            torch.empty(0, dtype=torch.long, device=logits.device),
            torch.empty(0, device=logits.device),
            torch.empty(0, logits.shape[0], dtype=torch.bool, device=logits.device),
        )

    mask_logits = torch.stack(logits_per_mask, dim=0)
    pred_scores, pred_classes = torch.max(mask_logits, dim=1)
    mask_tensor = torch.stack(mask_list, dim=0)
    return pred_classes, pred_scores, mask_tensor


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    torch.set_grad_enabled(False)

    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root)
    cfg = Config.fromfile(args.config)
    inferencer = LangPretrainerInference(cfg, args.checkpoint, device=args.device)

    eval_device_str = args.eval_device or str(inferencer.device)
    if eval_device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested for evaluation but not available; falling back to CPU."
        )
        eval_device_str = "cpu"
    eval_device = torch.device(eval_device_str)

    labels = read_class_labels(args.labels_file)
    subset_mapper = build_subset_mapper(labels)
    instance_ignore_idx = compute_instance_ignore_idx(labels)
    segment_ignore_index = sorted(set([-1] + instance_ignore_idx))

    text_embeddings = torch.load(
        args.text_embeddings, map_location="cpu", weights_only=True
    )
    if isinstance(text_embeddings, dict):
        raise ValueError("Expected a tensor in text embedding file, found dict.")
    if text_embeddings.shape[0] != len(labels):
        raise ValueError(
            f"Text embeddings ({text_embeddings.shape[0]}) do not match number of classes ({len(labels)})."
        )
    text_embeddings = F.normalize(text_embeddings.float(), p=2, dim=1).to(eval_device)

    evaluator = InstanceSegmentationEvaluator(
        class_names=labels,
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
        min_region_size=args.min_region_size,
        subset_mapper=subset_mapper,
    )

    feat_keys = inferencer.feat_keys
    scenes = discover_scenes(data_root, args.scenes_list, args.limit)
    if not scenes:
        raise RuntimeError("No scenes found to evaluate.")

    logger.info("Evaluating %d scenes (teacher=%s)", len(scenes), args.teacher_name)

    failed = []
    for idx, scene_name in enumerate(scenes, start=1):
        scene_dir = data_root / scene_name
        try:
            scene_inputs, segment, instance = load_scene_arrays(scene_dir, feat_keys)
            outputs = inferencer(scene_inputs, scene_name=scene_name, save=False)
            teacher_feats = outputs["teacher_features"].get(args.teacher_name)
            if teacher_feats is None:
                raise KeyError(
                    f"Teacher '{args.teacher_name}' features not found in inference output."
                )
            if isinstance(teacher_feats, np.ndarray):
                point_feat = torch.from_numpy(teacher_feats).to(eval_device)
            else:
                point_feat = teacher_feats.to(eval_device)
            point_feat = F.normalize(point_feat.float(), p=2, dim=1)
            if point_feat.shape[1] != text_embeddings.shape[1]:
                raise ValueError(
                    f"Feature dim {point_feat.shape[1]} does not match text embedding dim {text_embeddings.shape[1]}"
                )
            if point_feat.shape[0] != segment.shape[0]:
                raise ValueError(
                    f"Point count mismatch between features ({point_feat.shape[0]}) and labels ({segment.shape[0]})."
                )

            logits = torch.matmul(point_feat, text_embeddings.t())
            masks_binary = load_mask_binary(mask_root, scene_name)
            if masks_binary.shape[1] != logits.shape[0]:
                raise ValueError(
                    f"Mask columns ({masks_binary.shape[1]}) do not match number of points ({logits.shape[0]})."
                )
            mask_tensor = torch.from_numpy(masks_binary).to(eval_device)
            pred_classes, pred_scores, valid_masks = aggregate_masks(
                logits, mask_tensor, instance_ignore_idx
            )
            if pred_classes.numel() == 0:
                logger.warning(
                    "Scene %s has no valid masks after filtering; skipping.", scene_name
                )
                continue

            evaluator.update(
                pred_classes=pred_classes.cpu(),
                pred_scores=pred_scores.cpu(),
                pred_masks=valid_masks.cpu(),
                gt_segment=torch.from_numpy(segment).long(),
                gt_instance=torch.from_numpy(instance).long(),
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed on scene %s: %s", scene_name, exc)
            failed.append(scene_name)

        if idx % args.log_interval == 0:
            logger.info("Processed %d / %d scenes", idx, len(scenes))

    results = evaluator.compute()
    logger.info("Evaluation complete.")
    summary_keys = [
        "map",
        "map50",
        "map25",
        "map_head",
        "map_common",
        "map_tail",
    ]
    for key in summary_keys:
        if key in results:
            logger.info("%s: %.4f", key, results[key])

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved metrics to %s", save_path)

    if failed:
        logger.warning("Failed scenes (%d): %s", len(failed), ", ".join(failed))


if __name__ == "__main__":
    main()
