import os
import time
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from typing import Any, Dict, List, Optional, Tuple

from .test import TesterBase, TESTERS
from pointcept.datasets import collate_fn
from pointcept.utils.logger import get_root_logger
import pointcept.utils.comm as comm
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    make_dirs,
    neighbor_voting,
    clustering_voting,
)
from pointcept.engines.hooks.eval_helper import (
    ZeroShotSemSegEvaluatorHelper,
    FeatureSimilarityEvaluatorHelper,
)
from pointcept.utils.wandb_utils import is_wandb_active, safe_wandb_log


@TESTERS.register_module()
class LangPretrainMultiTeacherTester(TesterBase):
    """Tester supporting zero-shot lang evaluation plus additional teacher metrics."""

    def __init__(
        self,
        cfg,
        model=None,
        test_loader=None,
        verbose=True,
        teachers: Optional[List[Dict]] = None,
        evaluate_teachers: Optional[List[str]] = None,
        chunk_size: int = 600000,
        **kwargs,
    ) -> None:
        super().__init__(cfg, model, test_loader, verbose, **kwargs)
        if "index" in kwargs:
            cfg = copy.deepcopy(cfg)
            cfg["test"] = cfg["test"][kwargs["index"]]
            cfg.data.test = cfg.data.test[kwargs["index"]]
        self.cfg = cfg
        self.logger = get_root_logger()
        self.chunk_size = chunk_size

        test_cfg = cfg["test"]
        teachers_cfg = teachers if teachers is not None else test_cfg.get("teachers", [])
        if not teachers_cfg:
            raise ValueError("LangPretrainMultiTeacherTester requires teacher definitions")
        self.evaluate_teachers = (
            evaluate_teachers if evaluate_teachers is not None else test_cfg.get("evaluate_teachers")
        )

        self.teacher_info = {}
        self._build_teacher_helpers(teachers_cfg)

        if self.zero_shot_name is None:
            raise ValueError("At least one teacher with type='zero_shot' is required")

        # keep backward compatibility fields used within the original zero-shot pipeline
        zcfg = self.teacher_info[self.zero_shot_name]
        helper = zcfg["helper"]
        self.class_names = helper.class_names
        self.text_embeddings = helper.text_embeddings
        self.excluded_indices = helper.excluded_indices
        self.keep_indices = [i for i in range(helper.num_classes) if i not in self.excluded_indices]
        self.num_classes = helper.num_classes

        self.enable_voting = zcfg["config"]["enable_voting"]
        self.vote_k = zcfg["config"]["vote_k"]
        self.confidence_threshold = zcfg["config"]["confidence_threshold"]
        self.ignore_index = zcfg["config"]["ignore_index"]
        self.save_feat = zcfg["config"]["save_feat"]
        self.skip_eval = zcfg["config"]["skip_eval"]
        self.pred_label_mapping = zcfg["config"].get("pred_label_mapping")

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------
    def _active_teachers(self, override: Optional[List[str]] = None) -> List[str]:
        selection = override if override is not None else self.evaluate_teachers
        if selection is None:
            return list(self.teacher_info.keys())
        resolved = list(selection)
        for name in resolved:
            if name not in self.teacher_info:
                raise KeyError(f"Requested teacher '{name}' not found in tester configuration")
        return resolved

    def _build_teacher_helpers(self, teachers_cfg: List[Dict]):
        self.zero_shot_name = None
        for cfg in teachers_cfg:
            name = cfg.get("name")
            eval_type = cfg.get("type")
            if name is None or eval_type is None:
                raise ValueError("Teacher config must include 'name' and 'type'")
            if name in self.teacher_info:
                raise ValueError(f"Duplicate teacher name: {name}")
            eval_type = eval_type.lower()
            if eval_type == "zero_shot":
                helper = self._create_zero_shot_helper(cfg)
                if self.zero_shot_name is not None:
                    raise ValueError("Multiple zero-shot teachers are not supported")
                self.zero_shot_name = name
                self.teacher_info[name] = dict(
                    type="zero_shot",
                    helper=helper,
                    config=dict(
                        enable_voting=cfg.get("enable_voting", True),
                        vote_k=cfg.get("vote_k", 25),
                        confidence_threshold=cfg.get("confidence_threshold", 0.1),
                        ignore_index=cfg.get("ignore_index", -1),
                        save_feat=cfg.get("save_feat", False),
                        skip_eval=cfg.get("skip_eval", False),
                        pred_label_mapping=cfg.get("pred_label_mapping"),
                    ),
                )
                self.logger.info(f"Tester [{name}] config: {self.teacher_info[name]['config']}")
            elif eval_type == "feature_similarity":
                helper = FeatureSimilarityEvaluatorHelper(
                    target_key=cfg.get("target_key", f"{name}_feat"),
                    mask_key=cfg.get("mask_key"),
                    mask_min_norm=cfg.get("mask_min_norm", 0.05),
                    sample_stride=cfg.get("sample_stride", 4),
                    chunk_size=cfg.get("chunk_size", 200000),
                )
                self.teacher_info[name] = dict(
                    type="feature_similarity",
                    helper=helper,
                )
                self.logger.info(f"Tester [{name}] config: {self.teacher_info[name]['helper']}")
            else:
                raise ValueError(f"Unsupported teacher evaluation type: {eval_type}")

    def _create_zero_shot_helper(self, cfg: Dict) -> ZeroShotSemSegEvaluatorHelper:
        class_names_path = cfg.get("class_names")
        text_embeddings_path = cfg.get("text_embeddings")
        if class_names_path is None or text_embeddings_path is None:
            raise ValueError("Zero-shot teacher requires 'class_names' and 'text_embeddings'")

        with open(class_names_path, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
        text_embeddings = torch.load(text_embeddings_path, weights_only=True)
        return ZeroShotSemSegEvaluatorHelper(
            class_names=class_names,
            text_embeddings=text_embeddings,
            excluded_classes=cfg.get("excluded_classes"),
            ignore_index=cfg.get("ignore_index", -1),
            confidence_threshold=cfg.get("confidence_threshold", 0.1),
            vote_k=cfg.get("vote_k", 25),
            enable_voting=cfg.get("enable_voting", True),
            pred_label_mapping=cfg.get("pred_label_mapping"),
        )

    # ------------------------------------------------------------------
    # testing loop
    # ------------------------------------------------------------------
    def test(self):
        active_teachers = self._active_teachers()
        self._run_test(active_teachers)

    def test_teachers(self, teacher_names: List[str]) -> None:
        """Public helper to evaluate a subset of teachers with current weights."""
        active_teachers = self._active_teachers(teacher_names)
        self._run_test(active_teachers)

    def _run_test(self, active_teachers: List[str]) -> None:
        if not active_teachers:
            self.logger.warning(
                "LangPretrainMultiTeacherTester received empty evaluate_teachers; skipping test."
            )
            return

        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(
            ">>>>>>>>>>>>>> LangPretrainMultiTeacherTester Start Evaluation >>>>>>>>>>>>>"
        )
        logger.info(
            f"Testing on {self.cfg.data.test.split} split of {self.cfg.data.test.type}"
        )
        logger.info(f"Active teachers: {active_teachers}")

        zero_shot_requested = self.zero_shot_name in active_teachers
        zero_shot_metrics_enabled = zero_shot_requested and not self.skip_eval
        zero_shot_helper = self.teacher_info[self.zero_shot_name]["helper"]
        zero_shot_helper.reset()
        feature_helpers = {
            name: info["helper"]
            for name, info in self.teacher_info.items()
            if info["type"] == "feature_similarity"
        }
        for helper in feature_helpers.values():
            helper.reset()

        feature_teacher_helpers = {
            name: helper
            for name, helper in feature_helpers.items()
            if name in active_teachers
        }

        batch_time = AverageMeter()
        intersection_meter = AverageMeter() if zero_shot_metrics_enabled else None
        union_meter = AverageMeter() if zero_shot_metrics_enabled else None
        target_meter = AverageMeter() if zero_shot_metrics_enabled else None
        record: Dict[str, Dict[str, np.ndarray]] = {} if zero_shot_metrics_enabled else {}
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, f"result_{self.cfg.data.test.type}"
        )
        make_dirs(save_path)

        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
            or "ScanNetPP" in self.cfg.data.test.type
            or "ScanNet" in self.cfg.data.test.type
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        if self.save_feat and zero_shot_requested:
            make_dirs(os.path.join(save_path, "feat"))
        comm.synchronize()

        for idx, data_dict_raw in enumerate(self.test_loader):
            tic = time.time()
            data_dict = data_dict_raw[0]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment", None)

            dino_target = data_dict.pop("origin_dino_feat",  None)
            valid_feat_mask = data_dict.get("origin_feat_mask",  None)
            
            data_name = data_dict.pop("name", f"scene_{idx}")
            pred_save_path = os.path.join(save_path, f"{data_name}_pred.npy")
            feat_save_path = (
                os.path.join(save_path, "feat", f"{data_name}_feat.pth")
                if self.save_feat and zero_shot_requested
                else None
            )

            # reuse previous predictions if available and evaluation only
            reuse_pred = (
                zero_shot_requested
                and os.path.isfile(pred_save_path)
                and not self.save_feat
                and "pc_coord" not in data_dict
                and not feature_teacher_helpers
            )
            if reuse_pred:
                logger.info(f"{data_name}: loaded existing prediction")
                pred = np.load(pred_save_path)
                if "pc_segment" in data_dict and "pc_coord" in data_dict:
                    segment = data_dict["pc_segment"]
                elif "origin_segment" in data_dict:
                    segment = data_dict["origin_segment"]
                if "ScanNetPP" in self.cfg.data.test.type:
                    pred = pred[:, 0]
            else:
                coords = data_dict.get("coord")
                num_points = (
                    segment.size
                    if segment is not None
                    else (coords.shape[0] if coords is not None else 0)
                )

                pred_tensor = None
                pred_coord = None
                if zero_shot_requested:
                    num_classes = self.text_embeddings.size(0)
                    pred_tensor = torch.zeros(
                        (num_points, num_classes), device="cuda", dtype=torch.float16
                    )
                    pred_coord = torch.zeros((num_points, 3), device="cuda", dtype=torch.float16)

                accumulators: Dict[str, torch.Tensor] = {}
                counts: Dict[str, torch.Tensor] = {}

                for frag in fragment_list:
                    input_dict = collate_fn([frag])
                    for key in input_dict:
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)

                    idx_part = input_dict["index"]
                    offset_list = input_dict["offset"]

                    with torch.no_grad():
                        out_dict = self.model(input_dict, chunk_size=self.chunk_size)
                        point_feat_map = out_dict["point_feat"]

                    bs = 0
                    for be in offset_list:
                        slice_idx = idx_part[bs:be]

                        if zero_shot_requested and not self.skip_eval and pred_tensor is not None:
                            lang_feat = point_feat_map[self.zero_shot_name][bs:be]
                            logits = torch.mm(lang_feat, self.text_embeddings.t())
                            prob = torch.sigmoid(logits)
                            pred_tensor[slice_idx] += prob
                            pred_coord[slice_idx] = input_dict["coord"][bs:be]

                            if self.save_feat:
                                acc = accumulators.get(self.zero_shot_name)
                                cnt = counts.get(self.zero_shot_name)
                                if acc is None:
                                    feat_dim = lang_feat.size(1)
                                    accumulators[self.zero_shot_name] = torch.zeros(
                                        (num_points, feat_dim), device="cuda"
                                    )
                                    counts[self.zero_shot_name] = torch.zeros(
                                        num_points, device="cuda"
                                    )
                                    acc = accumulators[self.zero_shot_name]
                                    cnt = counts[self.zero_shot_name]
                                acc[slice_idx] += lang_feat
                                cnt[slice_idx] += 1

                        for name, helper in feature_teacher_helpers.items():
                            feat_chunk = point_feat_map[name][bs:be]
                            acc = accumulators.get(name)
                            cnt = counts.get(name)
                            if acc is None:
                                feat_dim = feat_chunk.size(1)
                                accumulators[name] = torch.zeros(
                                    (num_points, feat_dim), device="cuda"
                                )
                                counts[name] = torch.zeros(num_points, device="cuda")
                                acc = accumulators[name]
                                cnt = counts[name]
                            acc[slice_idx] += feat_chunk
                            cnt[slice_idx] += 1

                        bs = be

                if zero_shot_requested and pred_tensor is not None:
                    max_probs, argmax_indices = torch.max(pred_tensor, dim=1)
                    argmax_indices[max_probs < self.confidence_threshold] = self.ignore_index
                    pred = argmax_indices.cpu().numpy()
                else:
                    pred = None

                if (
                    self.save_feat
                    and zero_shot_requested
                    and self.zero_shot_name in accumulators
                ):
                    acc = accumulators[self.zero_shot_name]
                    cnt = counts[self.zero_shot_name]
                    mask = cnt > 0
                    acc[mask] /= cnt[mask].unsqueeze(1)
                    if "inverse" in data_dict:
                        acc = acc[data_dict["inverse"]]
                    acc = F.normalize(acc, p=2, dim=1)
                    torch.save(acc.cpu(), feat_save_path)
                    del accumulators[self.zero_shot_name]
                    del counts[self.zero_shot_name]
                    del acc

                for name, helper in feature_teacher_helpers.items():
                    acc = accumulators.get(name)
                    cnt = counts.get(name)
                    if acc is None:
                        continue
                    mask = cnt > 0
                    acc[mask] /= cnt[mask].unsqueeze(1)
                    if "inverse" in data_dict:
                        acc = acc[data_dict["inverse"]]
                    acc = F.normalize(acc, p=2, dim=1)

                    if dino_target is not None:
                        if not isinstance(dino_target, torch.Tensor):
                            target_tensor = torch.as_tensor(dino_target)
                        else:
                            target_tensor = dino_target
                        helper_input = {
                            helper.target_key: target_tensor.to(acc.device, dtype=acc.dtype)
                        }
                        if valid_feat_mask is not None:
                            mask_tensor = (
                                valid_feat_mask
                                if isinstance(valid_feat_mask, torch.Tensor)
                                else torch.as_tensor(valid_feat_mask)
                            )
                            helper_input["valid_feat_mask"] = mask_tensor.to(acc.device)
                        if segment is not None:
                            seg_tensor = (
                                segment
                                if isinstance(segment, torch.Tensor)
                                else torch.as_tensor(segment)
                            )
                            helper_input["segment"] = seg_tensor.to(acc.device)
                        helper.process_batch(helper_input, acc)
                        del helper_input
                    if name in accumulators:
                        del accumulators[name]
                    if name in counts:
                        del counts[name]
                    del acc

            # Neighbor voting & submission outputs
            if zero_shot_requested and pred is not None and self.enable_voting:
                num_classes = self.num_classes
                ignore_index = self.ignore_index
                if "pc_coord" in data_dict:
                    # pc_coord takes priority if available
                    coords = data_dict.get("origin_coord", data_dict["coord"])
                    if "origin_coord" in data_dict:
                        assert "inverse" in data_dict, (
                            "Inverse mapping is required to map pred to full origin_coord"
                        )
                        pred = pred[data_dict["inverse"]]  # shape => [original_num_points, ...]
                    query_coords = data_dict["pc_coord"]
                    pred = neighbor_voting(
                        coords,
                        pred,
                        self.vote_k,
                        ignore_index,
                        num_classes,
                        valid_mask=data_dict.get("origin_feat_mask"),
                        query_coords=query_coords,
                    )
                    # eval on pc segment labels
                    if "pc_segment" in data_dict:
                        segment = data_dict["pc_segment"]
                elif "origin_coord" in data_dict:
                    assert "inverse" in data_dict, (
                        "Inverse mapping is required to map pred to full origin_coord"
                    )
                    pred = pred[data_dict["inverse"]]  # shape => [original_num_points, ...]
                    coords = data_dict["origin_coord"]
                    pred = neighbor_voting(
                        coords,
                        pred,
                        self.vote_k,
                        ignore_index,
                        num_classes,
                        valid_mask=data_dict.get("origin_feat_mask"),
                    )
                    # eval on origin_segment
                    if "origin_segment" in data_dict:
                        segment = data_dict["origin_segment"]
                if "origin_instance" in data_dict:
                    pred = clustering_voting(
                        pred, data_dict["origin_instance"], ignore_index
                    )

            if zero_shot_requested and pred is not None and self.pred_label_mapping is not None:
                for src, dst in self.pred_label_mapping.items():
                    pred[pred == src] = dst

            if zero_shot_requested and pred is not None:
                np.save(pred_save_path, pred)

                if ("ScanNetPP" in self.cfg.data.test.type) and pred.ndim == 2:
                    np.savetxt(
                        os.path.join(save_path, "submit", f"{data_name}.txt"),
                        pred.astype(np.int32),
                        delimiter=",",
                        fmt="%d",
                    )
                    pred = pred[:, 0]
                elif self.cfg.data.test.type in [
                    "ScanNetGSDataset",
                    "ScanNet200GSDataset",
                ]:
                    if comm.is_main_process():
                        np.savetxt(
                            os.path.join(save_path, "submit", f"{data_name}.txt"),
                            self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                            fmt="%d",
                        )
                elif (
                    self.cfg.data.test.type == "SemanticKITTIDataset"
                    and comm.is_main_process()
                ):
                    sequence_name, frame_name = data_name.split("_")
                    os.makedirs(
                        os.path.join(
                            save_path,
                            "submit",
                            "sequences",
                            sequence_name,
                            "predictions",
                        ),
                        exist_ok=True,
                    )
                    submit = pred.astype(np.uint32)
                    submit = np.vectorize(
                        self.test_loader.dataset.learning_map_inv.__getitem__
                    )(submit).astype(np.uint32)
                    submit.tofile(
                        os.path.join(
                            save_path,
                            "submit",
                            "sequences",
                            sequence_name,
                            "predictions",
                            f"{frame_name}.label",
                        )
                    )
                elif (
                    self.cfg.data.test.type == "NuScenesDataset"
                    and comm.is_main_process()
                ):
                    np.array(pred + 1).astype(np.uint8).tofile(
                        os.path.join(
                            save_path,
                            "submit",
                            "lidarseg",
                            "test",
                            f"{data_name}_lidarseg.bin",
                        )
                    )

            if zero_shot_metrics_enabled and pred is not None:
                intersection, union, target = intersection_and_union(
                    pred, segment, self.num_classes, self.ignore_index
                )
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)
                record[data_name] = dict(
                    intersection=intersection, union=union, target=target
                )

            batch_time.update(time.time() - tic)
            logger.info(
                f"Test scene {idx + 1}/{len(self.test_loader)} {data_name} "
                f"time {batch_time.val:.2f}s ({batch_time.avg:.2f}s avg)"
            )

        zero_shot_metrics = {}
        per_class_rows: List[List] = []
        if zero_shot_metrics_enabled:
            comm.synchronize()
            record_sync = comm.gather(record, dst=0)
            if comm.is_main_process():
                final_record = {}
                for part in record_sync:
                    final_record.update(part)
                zero_shot_metrics, per_class_rows = self._aggregate_zero_shot_results(
                    final_record
                )
                self.teacher_info[self.zero_shot_name]["metrics"] = zero_shot_metrics
                self._log_zero_shot_results(zero_shot_metrics, per_class_rows)
        else:
            comm.synchronize()

        feature_metrics = self._collect_feature_metrics(
            feature_teacher_helpers, active_teachers
        )
        if comm.is_main_process():
            self._log_feature_metrics(feature_metrics)

        logger.info(
            "<<<<<<<<<<<<<< LangPretrainMultiTeacherTester Finished <<<<<<<<<<<<<<"
        )


    def _aggregate_zero_shot_results(
        self, final_record: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, Any], List[List]]:
        if not final_record:
            zeros = np.zeros(self.num_classes, dtype=np.float64)
            intersection = zeros.copy()
            union = zeros.copy()
            target = zeros.copy()
        else:
            intersection = np.sum(
                [v["intersection"] for v in final_record.values()], axis=0
            )
            union = np.sum([v["union"] for v in final_record.values()], axis=0)
            target = np.sum([v["target"] for v in final_record.values()], axis=0)

        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)

        mask_present = target != 0
        mean_iou = float(np.mean(iou_class[mask_present])) if mask_present.any() else 0.0
        mean_acc = float(np.mean(accuracy_class[mask_present])) if mask_present.any() else 0.0
        overall_acc = float(intersection.sum() / (target.sum() + 1e-10))

        metrics: Dict[str, Any] = dict(
            mIoU=mean_iou, mAcc=mean_acc, allAcc=overall_acc
        )

        if self.excluded_indices:
            fg_iou = iou_class[self.keep_indices]
            fg_acc = accuracy_class[self.keep_indices]

            fg_mask_present = target[self.keep_indices] != 0
            fg_mIoU = float(np.mean(fg_iou[fg_mask_present])) if fg_mask_present.any() else 0.0
            fg_mAcc = float(np.mean(fg_acc[fg_mask_present])) if fg_mask_present.any() else 0.0
            fg_intersection = intersection[self.keep_indices]
            fg_target = target[self.keep_indices]
            fg_allAcc = float(fg_intersection.sum() / (fg_target.sum() + 1e-10))
            metrics.update(
                fg_mIoU=fg_mIoU, fg_mAcc=fg_mAcc, fg_allAcc=fg_allAcc
            )

        per_class_rows: List[List] = []
        for idx in range(self.num_classes):
            if self.class_names:
                cls_name = self.class_names[idx]
                label = f"Class_{idx}-{cls_name}"
            else:
                cls_name = f"Class_{idx}"
                label = cls_name
            per_class_rows.append(
                [
                    idx,
                    label,
                    float(iou_class[idx]),
                    float(accuracy_class[idx]),
                    "excluded" if idx in self.excluded_indices else "included",
                    "presented" if target[idx] > 0 else "absent",
                ]
            )

        metrics.update(
            iou_class=iou_class,
            accuracy_class=accuracy_class,
            intersection=intersection,
            union=union,
            target=target,
        )
        return metrics, per_class_rows

    def _log_zero_shot_results(
        self, metrics: Dict[str, Any], per_class_rows: List[List]
    ) -> None:
        logger = self.logger
        logger.info(
            f"Zero-shot [{self.zero_shot_name}] mIoU {metrics['mIoU']:.4f} "
            f"mAcc {metrics['mAcc']:.4f} overallAcc {metrics['allAcc']:.4f}"
        )
        if "fg_mIoU" in metrics:
            logger.info(
                "Foreground Val result (excluding {} classes): mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    len(self.excluded_indices),
                    metrics["fg_mIoU"],
                    metrics["fg_mAcc"],
                    metrics["fg_allAcc"],
                )
            )
        # log missing classes
        missing_classes = [row for row in per_class_rows if row[5] == "absent"]
        if missing_classes:
            missing_str = ", ".join([str(row[1]) for row in missing_classes])
            logger.info(f"Missing {len(missing_classes)} classes: ({missing_str})")

        for row in per_class_rows:
            if row[5] == "absent":
                continue
            _, label, iou, acc, _, _ = row
            logger.info(f"{label} Result: iou/accuracy {iou:.4f}/{acc:.4f}")

        if not getattr(self.cfg, "enable_wandb", False) or not is_wandb_active():
            return

        # wandb logging
        dataset_name = self.cfg.data.test.type
        wandb_payload = {
            f"Eval/{dataset_name}/mIoU": metrics["mIoU"],
            f"Eval/{dataset_name}/mAcc": metrics["mAcc"],
            f"Eval/{dataset_name}/allAcc": metrics["allAcc"],
        }

        if "fg_mIoU" in metrics:
            wandb_payload.update(
                {
                    f"Eval/{dataset_name}/fg_mIoU": metrics["fg_mIoU"],
                    f"Eval/{dataset_name}/fg_mAcc": metrics["fg_mAcc"],
                    f"Eval/{dataset_name}/fg_allAcc": metrics["fg_allAcc"],
                }
            )

        if per_class_rows:
            table = wandb.Table(
                columns=[
                    "Class_ID",
                    "Class_Label",
                    "IoU",
                    "Accuracy",
                    "Status",
                    "Presented",
                ],
                data=per_class_rows,
            )
            wandb_payload[f"Eval/{dataset_name}/per_class_results"] = table

        safe_wandb_log(wandb_payload, logger=logger)

    def _collect_feature_metrics(
        self,
        feature_helpers: Dict[str, FeatureSimilarityEvaluatorHelper],
        active_teachers: List[str],
    ) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for name, helper in feature_helpers.items():
            if name not in active_teachers:
                continue
            stats = dict(
                sum_cosine=helper.sum_cosine,
                sum_l2=helper.sum_l2,
                count=helper.count,
            )
            gathered = comm.gather(stats, dst=0)
            if not comm.is_main_process():
                continue
            total_cosine = 0.0
            total_l2 = 0.0
            total_count = 0
            for item in gathered:
                total_cosine += float(item.get("sum_cosine", 0.0))
                total_l2 += float(item.get("sum_l2", 0.0))
                total_count += int(item.get("count", 0))
            if total_count == 0:
                metrics[name] = {
                    "cosine_similarity": 0.0,
                    "cosine_loss": 1.0,
                    "l2_loss": 0.0,
                }
            else:
                mean_cos = total_cosine / total_count
                mean_l2 = total_l2 / total_count
                metrics[name] = {
                    "cosine_similarity": float(mean_cos),
                    "cosine_loss": float(1.0 - mean_cos),
                    "l2_loss": float(mean_l2),
                }
        return metrics

    def _log_feature_metrics(self, metrics_map: Dict[str, Dict[str, float]]) -> None:
        if not metrics_map:
            return

        dataset_name = self.cfg.data.test.type
        wandb_enabled = getattr(self.cfg, "enable_wandb", False)
        wandb_payload: Dict[str, float] = {}

        for name, metrics in metrics_map.items():
            self.teacher_info[name]["metrics"] = metrics
            self.logger.info(
                f"Feature similarity [{name}] cosine_sim {metrics['cosine_similarity']:.4f} "
                f"cosine_loss {metrics['cosine_loss']:.4f} l2_loss {metrics['l2_loss']:.4f}"
            )
            if not wandb_enabled:
                continue
            prefix = f"Eval/{dataset_name}/{name}"
            wandb_payload.update(
                {
                    f"{prefix}/cosine_similarity": metrics["cosine_similarity"],
                    f"{prefix}/cosine_loss": metrics["cosine_loss"],
                    f"{prefix}/l2_loss": metrics["l2_loss"],
                }
            )

        if wandb_enabled and wandb_payload and is_wandb_active():
            safe_wandb_log(wandb_payload, logger=self.logger)


    @staticmethod
    def collate_fn(batch):
        return batch
