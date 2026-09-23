import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.distributed as dist
import torch.nn.functional as F
import gc

from uuid import uuid4
from typing import Dict, List, Optional, Tuple

import pointcept.utils.comm as comm
from pointcept.utils.wandb_utils import (
    safe_wandb_define_metric,
    safe_wandb_log,
)
from pointcept.utils.misc import (
    clustering_voting,
    _majority_vote,
    inspect_memory,
)

from .default import HookBase
from .builder import HOOKS
from .eval_helper import (ZeroShotEvalResult, 
                          ZeroShotSemSegEvaluatorHelper, 
                          FeatureSimilarityEvaluatorHelper
)


class _LangPretrainMultiTeacherEvalBase(HookBase):
    def __init__(self, teachers, evaluate_teachers=None, chunk_size=600000):
        super().__init__()
        if not teachers:
            raise ValueError("'teachers' configuration must not be empty")
        self.raw_teacher_cfgs = teachers
        self.evaluate_teachers = evaluate_teachers
        self.chunk_size = chunk_size
        self.teacher_info = {}
        self._build_teacher_helpers()

    @staticmethod
    def _load_class_names(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _build_teacher_helpers(self):
        for cfg in self.raw_teacher_cfgs:
            teacher_name = cfg.get("name")
            eval_type = cfg.get("type").lower()
            if teacher_name is None or eval_type is None:
                raise ValueError("Each eval config must include the teacher name and eval_type")
            if teacher_name in self.teacher_info:
                raise ValueError(f"Duplicate teacher name provided: {teacher_name}")

            select_metric = cfg.get(
                "select_metric",
                "mIoU" if eval_type == "zero_shot" else "cosine_similarity",
            )
            log_prefix = cfg.get("log_prefix", teacher_name)

            if eval_type == "zero_shot":
                helpers = self._build_zero_shot_helpers(cfg)
            elif eval_type == "feature_similarity":
                helpers = [
                    FeatureSimilarityEvaluatorHelper(
                        target_key=cfg.get("target_key", f"{teacher_name}_feat"),
                        mask_key=cfg.get("mask_key"),
                        mask_min_norm=cfg.get("mask_min_norm", 0.05),
                        sample_stride=cfg.get("sample_stride", 4),
                        chunk_size=cfg.get("chunk_size", 200_000),
                    )
                ]
            else:
                raise ValueError(f"Unsupported teacher eval type: {eval_type}")

            self.teacher_info[teacher_name] = dict(
                type=eval_type,
                helpers=helpers,
                select_metric=select_metric,
                log_prefix=log_prefix,
                cfg=cfg,
            )

    def _build_zero_shot_helpers(self, cfg):
        class_names_cfg = cfg.get("class_names")
        text_embeddings_cfg = cfg.get("text_embeddings")
        if class_names_cfg is None or text_embeddings_cfg is None:
            raise ValueError(
                "Zero-shot teacher requires 'class_names' and 'text_embeddings'"
            )

        excluded_cfg = cfg.get("excluded_classes")
        pred_map_cfg = cfg.get("pred_label_mapping")
        ignore_index = cfg.get("ignore_index", -1)
        confidence_threshold = cfg.get("confidence_threshold", 0.1)
        vote_k = cfg.get("vote_k", 25)
        enable_voting = cfg.get("enable_voting", True)

        helpers = []
        if isinstance(class_names_cfg, (list, tuple)):
            if not isinstance(text_embeddings_cfg, (list, tuple)):
                raise ValueError(
                    "When 'class_names' is a list, 'text_embeddings' must also be a list"
                )
            if len(class_names_cfg) != len(text_embeddings_cfg):
                raise ValueError(
                    "Mismatched lengths between class_names and text_embeddings lists"
                )
            for idx, class_name_path in enumerate(class_names_cfg):
                class_names = self._load_class_names(class_name_path)
                text_embeddings = torch.load(text_embeddings_cfg[idx], weights_only=True)
                excluded = excluded_cfg[idx] if excluded_cfg else None
                pred_map = pred_map_cfg[idx] if pred_map_cfg else None
                helpers.append(
                    ZeroShotSemSegEvaluatorHelper(
                        class_names=class_names,
                        text_embeddings=text_embeddings,
                        excluded_classes=excluded,
                        ignore_index=ignore_index,
                        confidence_threshold=confidence_threshold,
                        vote_k=vote_k,
                        enable_voting=enable_voting,
                        pred_label_mapping=pred_map,
                    )
                )
        else:
            class_names = self._load_class_names(class_names_cfg)
            text_embeddings = torch.load(text_embeddings_cfg, weights_only=True)
            helpers.append(
                ZeroShotSemSegEvaluatorHelper(
                    class_names=class_names,
                    text_embeddings=text_embeddings,
                    excluded_classes=excluded_cfg,
                    ignore_index=ignore_index,
                    confidence_threshold=confidence_threshold,
                    vote_k=vote_k,
                    enable_voting=enable_voting,
                    pred_label_mapping=pred_map_cfg,
                )
            )
        return helpers

    def _active_teachers(self):
        if self.evaluate_teachers is None:
            return list(self.teacher_info.keys())
        active = []
        for name in self.evaluate_teachers:
            if name not in self.teacher_info:
                raise KeyError(f"Requested teacher '{name}' not found in configuration")
            active.append(name)
        return active

    def _get_helper_for_dataset(self, teacher_name, dataset_idx=0):
        helpers = self.teacher_info[teacher_name]["helpers"]
        if dataset_idx < len(helpers):
            return helpers[dataset_idx]
        return helpers[-1]

    @staticmethod
    def _extract_teacher_feature(point_feat_dict, teacher_name):
        if not isinstance(point_feat_dict, dict):
            raise TypeError("Model output for point_feat must be a dict for multi-teacher eval")
        if teacher_name not in point_feat_dict:
            available = ", ".join(point_feat_dict.keys())
            raise KeyError(
                f"Teacher '{teacher_name}' not found in model outputs. Available keys: {available}"
            )
        return point_feat_dict[teacher_name]

    def _log_zero_shot_result(self, teacher_name, result: ZeroShotEvalResult, metrics_prefix: str, helper_idx=0):
        metrics = result.metrics
        helper = self.teacher_info[teacher_name]["helpers"][helper_idx]
        class_names = helper.class_names

        self.trainer.logger.info(
            f"[{teacher_name}] Missing classes: {result.missing_classes}"
        )
        self.trainer.logger.info(
            f"[{teacher_name}] --- Per-class IoU (all present classes) ---"
        )
        for c in result.present_classes:
            self.trainer.logger.info(
                f"[{teacher_name}] {class_names[c]:20s}: {result.ious[c]:.4f}"
            )
            if self.trainer.cfg.enable_wandb:
                safe_wandb_log(
                    {f"zero_shot/{metrics_prefix}/cls_{c}-{class_names[c]} IoU": result.ious[c]},
                    logger=self.trainer.logger,
                )

        self.trainer.logger.info(
            f"[{teacher_name}] Global Accuracy : {metrics['global_acc']:.4f}"
        )
        self.trainer.logger.info(
            f"[{teacher_name}] Mean Class Acc : {metrics['mean_class_acc']:.4f}"
        )
        self.trainer.logger.info(
            f"[{teacher_name}] Mean IoU       : {metrics['mIoU']:.4f}"
        )
        self.trainer.logger.info(
            f"[{teacher_name}] Foreground mIoU : {metrics['fg_mIoU']:.4f}"
        )
        self.trainer.logger.info(
            f"[{teacher_name}] Foreground mAcc : {metrics['fg_mAcc']:.4f}"
        )

    def _log_zero_shot_tensorboard(self, teacher_name, metrics_prefix, metrics):
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                f"{metrics_prefix}/mIoU", metrics["mIoU"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"{metrics_prefix}/fg_mIoU", metrics["fg_mIoU"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"{metrics_prefix}/global_acc", metrics["global_acc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"{metrics_prefix}/mean_class_acc",
                metrics["mean_class_acc"],
                current_epoch,
            )
            self.trainer.writer.add_scalar(
                f"{metrics_prefix}/fg_mAcc", metrics["fg_mAcc"], current_epoch
            )
            if self.trainer.cfg.enable_wandb:
                safe_wandb_log(
                    {
                        "Epoch": current_epoch,
                        f"{metrics_prefix}/mIoU": metrics["mIoU"],
                        f"{metrics_prefix}/fg_mIoU": metrics["fg_mIoU"],
                        f"{metrics_prefix}/global_acc": metrics["global_acc"],
                        f"{metrics_prefix}/mAcc": metrics["mean_class_acc"],
                        f"{metrics_prefix}/fg_mAcc": metrics["fg_mAcc"],
                    },
                    logger=self.trainer.logger,
                )

    def _log_feature_similarity(self, teacher_name, metrics_prefix, metrics):
        self.trainer.logger.info(
            f"[{teacher_name}] Cosine similarity: {metrics['cosine_similarity']:.4f}"
        )
        self.trainer.logger.info(
            f"[{teacher_name}] Cosine loss       : {metrics['cosine_loss']:.4f}"
        )
        self.trainer.logger.info(
            f"[{teacher_name}] L2 loss          : {metrics['l2_loss']:.4f}"
        )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                f"{metrics_prefix}/cosine_similarity",
                metrics["cosine_similarity"],
                current_epoch,
            )
            self.trainer.writer.add_scalar(
                f"{metrics_prefix}/cosine_loss",
                metrics["cosine_loss"],
                current_epoch,
            )
            self.trainer.writer.add_scalar(
                f"{metrics_prefix}/l2_loss",
                metrics["l2_loss"],
                current_epoch,
            )
            if self.trainer.cfg.enable_wandb:
                safe_wandb_log(
                    {
                        "Epoch": current_epoch,
                        f"{metrics_prefix}/cosine_similarity": metrics[
                            "cosine_similarity"
                        ],
                        f"{metrics_prefix}/cosine_loss": metrics["cosine_loss"],
                        f"{metrics_prefix}/l2_loss": metrics["l2_loss"],
                    },
                    logger=self.trainer.logger,
                )


@HOOKS.register_module()
class LangPretrainZeroShotSemSegEval(HookBase):
    def __init__(
        self,
        class_names,
        text_embeddings,
        excluded_classes=None,
        ignore_index=-1,
        confidence_threshold=0.1,
        vote_k=25,
        enable_voting=True,
        pred_label_mapping=None,
    ):
        """
        Args:
            class_names (list): path to a txt of class names ordered by class index
            text_embeddings (Tensor): path to text embeddings (num_classes, feat_dim)
            excluded_classes (list): Class names to exclude from final metrics
            ignore_index (int): Index to ignore in GT labels
            confidence_threshold (float): Minimum confidence to consider prediction valid
        """
        super().__init__()
        with open(class_names, "r") as f:
            class_name_list = [line.strip() for line in f if line.strip()]

        print(f"Loading text embeddings from {text_embeddings}")
        text_embeddings_tensor = torch.load(text_embeddings, weights_only=True)
        print(
            f"Text embeddings for ZeroShotSemSegEval with shape: {text_embeddings_tensor.shape}"
        )

        self.helper = ZeroShotSemSegEvaluatorHelper(
            class_names=class_name_list,
            text_embeddings=text_embeddings_tensor,
            excluded_classes=excluded_classes,
            ignore_index=ignore_index,
            confidence_threshold=confidence_threshold,
            vote_k=vote_k,
            enable_voting=enable_voting,
            pred_label_mapping=pred_label_mapping,
        )

        self.enable_voting = enable_voting
        self.vote_k = vote_k
        self.confidence_threshold = confidence_threshold
        self.ignore_index = ignore_index
        self.chunk_size = 600000

    def _reset_metrics(self):
        self.helper.reset()

    def _extract_point_features(self, output_dict):
        point_feat = output_dict.get("point_feat")
        if isinstance(point_feat, dict):
            if "feat" in point_feat:
                return point_feat["feat"]
            # fallback if model already returns teacher-specific dict
            if len(point_feat) == 1:
                return next(iter(point_feat.values()))
        raise KeyError("Unable to locate point features in model output")

    # def after_step(self): # for debug
    #     if self.trainer.cfg.evaluate:
    #         self.eval()
    #         self.trainer.model.train()

    def before_train(self):
        if self.trainer.cfg.enable_wandb:
            safe_wandb_define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Zero-Shot SemSeg Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        self._reset_metrics()

        import psutil

        process = psutil.Process()
        self.trainer.logger.info(
            f"Memory usage before eval: {process.memory_info().rss / 1024**3:.2f} GB"
        )

        if self.vote_k > 1 and self.enable_voting:
            self.trainer.logger.info(f"Neighbor voting enabled with k={self.vote_k}")

        helper = self.helper
        device = helper.device

        with torch.no_grad():
            self.trainer.logger.info(
                f"Length of val_loader: {len(self.trainer.val_loader)}"
            )
            for i, input_dict in enumerate(self.trainer.val_loader):
                input_dict = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in input_dict.items()
                }

                output_dict = self.trainer.model(input_dict, chunk_size=self.chunk_size)
                point_feat = self._extract_point_features(output_dict)
                helper.process_batch(input_dict, point_feat)

                if (i + 1) % 10 == 0:
                    self.trainer.logger.info(
                        f"Processed {i + 1}/{len(self.trainer.val_loader)} batches"
                    )

                del output_dict, point_feat, input_dict
                gc.collect()

        helper.synchronize()
        result = helper.compute_metrics()
        self._log_metrics(result)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.logger.info(
            f"Memory usage after eval: {process.memory_info().rss / 1024**3:.2f} GB"
        )
        gc.collect()
        torch.cuda.empty_cache()

    def _log_metrics(self, result: ZeroShotEvalResult):
        helper = self.helper
        metrics = result.metrics
        class_names = helper.class_names

        self.trainer.logger.info("\nMissing classes: %s", result.missing_classes)
        self.trainer.logger.info("\n--- Per-class IoU (all classes) ---")
        for c in result.present_classes:
            self.trainer.logger.info(f"{class_names[c]:20s}: {result.ious[c]:.4f}")
            if self.trainer.cfg.enable_wandb:
                safe_wandb_log(
                    {f"val/cls_{c}-{class_names[c]} IoU": result.ious[c]},
                    logger=self.trainer.logger,
                )

        self.trainer.logger.info("\n--- Metrics (ALL present classes) ---")
        self.trainer.logger.info(f"Global Accuracy   : {metrics['global_acc']:.4f}")
        self.trainer.logger.info(f"Mean Class Acc.   : {metrics['mean_class_acc']:.4f}")
        self.trainer.logger.info(f"Mean IoU (mIoU)   : {metrics['mIoU']:.4f}")

        self.trainer.logger.info(
            f"\n--- Foreground Metrics (EXCLUDED {helper.excluded_classes}) ---"
        )
        self.trainer.logger.info(f"Foreground mIoU   : {metrics['fg_mIoU']:.4f}")
        self.trainer.logger.info(f"Foreground mAcc   : {metrics['fg_mAcc']:.4f}")

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/mIoU", metrics["mIoU"], current_epoch)
            self.trainer.writer.add_scalar(
                "val/fg_mIoU", metrics["fg_mIoU"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/global_acc", metrics["global_acc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/mean_class_acc", metrics["mean_class_acc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/fg_mAcc", metrics["fg_mAcc"], current_epoch
            )
            if self.trainer.cfg.enable_wandb:
                safe_wandb_log(
                    {
                        "Epoch": current_epoch,
                        "val/mIoU": metrics["mIoU"],
                        "val/fg_mIoU": metrics["fg_mIoU"],
                        "val/global_acc": metrics["global_acc"],
                        "val/mAcc": metrics["mean_class_acc"],
                        "val/fg_mAcc": metrics["fg_mAcc"],
                    },
                    logger=self.trainer.logger,
                )

        self.trainer.comm_info["current_metric_value"] = metrics["fg_mIoU"]
        self.trainer.comm_info["current_metric_name"] = "fg_mIoU"


    def after_train(self):
        # self.enable_voting = True
        # self.eval()
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class LangPretrainMultiTeacherEval(_LangPretrainMultiTeacherEvalBase):
    def before_train(self):
        if self.trainer.cfg.enable_wandb:
            for teacher_name in self.teacher_info.keys():
                safe_wandb_define_metric(
                    f"val/{teacher_name}/*", step_metric="Epoch"
                )

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    # def after_step(self): # for debug
    #     if self.trainer.cfg.evaluate:
    #         self.eval()
    #         self.trainer.model.train()

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Multi-Teacher Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        import psutil
        process = psutil.Process()
        self.trainer.logger.info(
            f"Memory usage before eval: {process.memory_info().rss / 1024**3:.2f} GB"
        )

        active_teachers = self._active_teachers()
        for name in active_teachers:
            helper = self._get_helper_for_dataset(name, 0)
            helper.reset()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        val_loader = self.trainer.val_loader
        if isinstance(val_loader, (list, tuple)):
            if len(val_loader) != 1:
                raise TypeError(
                    "LangPretrainMultiTeacherEval expects a single validation loader; use LangPretrainMultiTeacherMultiEval for multiple loaders"
                )
            val_loader = val_loader[0]

        with torch.no_grad():
            for i, input_dict in enumerate(val_loader):
                input_dict = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in input_dict.items()
                }
                output_dict = self.trainer.model(input_dict, chunk_size=self.chunk_size)
                point_feat_map = output_dict.get("point_feat", {})
                for teacher_name in active_teachers:
                    helper = self._get_helper_for_dataset(teacher_name, 0)
                    feat = self._extract_teacher_feature(point_feat_map, teacher_name)
                    helper.process_batch(input_dict, feat)

                if (i + 1) % 10 == 0:
                    self.trainer.logger.info(
                        f"Processed {i + 1}/{len(val_loader)} batches"
                    )
                # free up memory
                del input_dict, output_dict, point_feat_map
                gc.collect()

        current_metrics = {}
        for teacher_name in active_teachers:
            info = self.teacher_info[teacher_name]
            helper = info["helpers"][0]
            metrics_prefix = f"val/{teacher_name}"
            if info["type"] == "zero_shot":
                helper.synchronize()
                result = helper.compute_metrics()
                self._log_zero_shot_result(teacher_name, result, metrics_prefix)
                self._log_zero_shot_tensorboard(
                    teacher_name, metrics_prefix, result.metrics
                )
                metric_value = result.metrics.get(info["select_metric"])
                if metric_value is None:
                    raise KeyError(
                        f"Metric '{info['select_metric']}' not found for teacher {teacher_name}"
                    )
                current_metrics[teacher_name] = dict(
                    name=info["select_metric"],
                    value=float(metric_value),
                )
            else:
                metrics = helper.compute_metrics()
                self._log_feature_similarity(teacher_name, metrics_prefix, metrics)
                metric_value = metrics.get(info["select_metric"])
                if metric_value is None:
                    raise KeyError(
                        f"Metric '{info['select_metric']}' not found for teacher {teacher_name}"
                    )
                current_metrics[teacher_name] = dict(
                    name=info["select_metric"],
                    value=float(metric_value),
                )

        first_teacher = active_teachers[0]
        self.trainer.comm_info["current_metric_value"] = current_metrics[first_teacher][
            "value"
        ]
        self.trainer.comm_info["current_metric_name"] = current_metrics[first_teacher][
            "name"
        ]
        # "current_metrics" to tell it's in multi-teacher eval mode
        self.trainer.comm_info["current_metrics"] = current_metrics

        self.trainer.logger.info(
            f"Memory usage after eval: {process.memory_info().rss / 1024**3:.2f} GB"
        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        gc.collect()
        torch.cuda.empty_cache()

    def after_train(self):
        if hasattr(self.trainer, "best_metrics") and self.trainer.best_metrics:
            summary = ", ".join(
                f"{k}:{v['value']:.4f}" for k, v in self.trainer.best_metrics.items()
            )
            self.trainer.logger.info(f"Best metrics per teacher -> {summary}")


@HOOKS.register_module()
class LangPretrainMultiTeacherMultiEval(_LangPretrainMultiTeacherEvalBase):
    def before_train(self):
        if self.trainer.cfg.enable_wandb:
            for teacher_name in self.teacher_info.keys():
                safe_wandb_define_metric(
                    f"val/{teacher_name}/*", step_metric="Epoch"
                )

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    # def after_step(self): # for debug
    #     if self.trainer.cfg.evaluate:
    #         self.eval()
    #         self.trainer.model.train()

    def eval(self):
        val_loaders = self.trainer.val_loader
        if not isinstance(val_loaders, (list, tuple)):
            raise TypeError(
                "LangPretrainMultiTeacherMultiEval expects trainer.val_loader to be a list of DataLoaders"
            )

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Multi-Teacher Multi-Data Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        import psutil

        process = psutil.Process()
        self.trainer.logger.info(
            f"Memory usage before eval: {process.memory_info().rss / 1024**3:.2f} GB"
        )

        active_teachers = self._active_teachers()
        per_teacher_values = {name: [] for name in active_teachers}

        for dataset_idx, loader in enumerate(val_loaders):
            dataset_obj = loader.dataset
            # unwrap nested datasets (e.g., Subset) until we reach the underlying dataset
            while hasattr(dataset_obj, "dataset"):
                dataset_obj = dataset_obj.dataset

            dataset_name = type(dataset_obj).__name__
            self.trainer.logger.info(
                f"Evaluating {dataset_name}, eval progress {dataset_idx + 1}/{len(val_loaders)}"
            )
            for teacher_name in active_teachers:
                helper = self._get_helper_for_dataset(teacher_name, dataset_idx)
                helper.reset()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                for i, input_dict in enumerate(loader):
                    input_dict = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in input_dict.items()
                    }
                    output_dict = self.trainer.model(
                        input_dict, chunk_size=self.chunk_size
                    )
                    point_feat_map = output_dict.get("point_feat", {})
                    for teacher_name in active_teachers:
                        helper = self._get_helper_for_dataset(
                            teacher_name, dataset_idx
                        )
                        feat = self._extract_teacher_feature(point_feat_map, teacher_name)
                        helper.process_batch(input_dict, feat)

                    if (i + 1) % 10 == 0:
                        self.trainer.logger.info(
                            f"  Processed {i + 1}/{len(loader)} batches"
                        )

                    del output_dict
                    gc.collect()

            for teacher_name in active_teachers:
                info = self.teacher_info[teacher_name]
                helper = self._get_helper_for_dataset(teacher_name, dataset_idx)
                metrics_prefix = f"val/{teacher_name}/data_{dataset_idx}"
                if info["type"] == "zero_shot":
                    helper.synchronize()
                    result = helper.compute_metrics()
                    self._log_zero_shot_result(teacher_name, result, f"{teacher_name}/data_{dataset_idx}", dataset_idx)
                    self._log_zero_shot_tensorboard(
                        teacher_name, metrics_prefix, result.metrics
                    )
                    metric_value = result.metrics.get(info["select_metric"])
                    if metric_value is None:
                        raise KeyError(
                            f"Metric '{info['select_metric']}' not found for teacher {teacher_name}"
                        )
                    per_teacher_values[teacher_name].append(float(metric_value))
                else:
                    metrics = helper.compute_metrics()
                    self._log_feature_similarity(teacher_name, metrics_prefix, metrics)
                    metric_value = metrics.get(info["select_metric"])
                    if metric_value is None:
                        raise KeyError(
                            f"Metric '{info['select_metric']}' not found for teacher {teacher_name}"
                        )
                    per_teacher_values[teacher_name].append(float(metric_value))

        current_metrics = {}
        for teacher_name in active_teachers:
            values = per_teacher_values[teacher_name]
            avg_value = float(sum(values) / len(values)) if values else 0.0
            current_metrics[teacher_name] = dict(
                name=self.teacher_info[teacher_name]["select_metric"],
                value=avg_value,
            )
            metrics_prefix = f"val/{teacher_name}"
            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar(
                    f"{metrics_prefix}/avg_{self.teacher_info[teacher_name]['select_metric']}",
                    avg_value,
                    current_epoch,
                )
            if self.trainer.cfg.enable_wandb:
                safe_wandb_log(
                    {
                        "Epoch": current_epoch,
                        f"{metrics_prefix}/avg_{self.teacher_info[teacher_name]['select_metric']}": avg_value,
                    },
                    logger=self.trainer.logger,
                )

        first_teacher = active_teachers[0]
        self.trainer.comm_info["current_metric_value"] = current_metrics[first_teacher][
            "value"
        ]
        self.trainer.comm_info["current_metric_name"] = current_metrics[first_teacher][
            "name"
        ]
        self.trainer.comm_info["current_metrics"] = current_metrics

        self.trainer.logger.info(
            f"Memory usage after eval: {process.memory_info().rss / 1024**3:.2f} GB"
        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        gc.collect()
        torch.cuda.empty_cache()

    def after_train(self):
        if hasattr(self.trainer, "best_metrics") and self.trainer.best_metrics:
            summary = ", ".join(
                f"{k}:{v['value']:.4f}" for k, v in self.trainer.best_metrics.items()
            )
            self.trainer.logger.info(f"Best metrics per teacher -> {summary}")
            
@HOOKS.register_module()
class LangPretrainZeroShotSemSeglMultiEval(HookBase):
    def __init__(
        self,
        class_names,
        text_embeddings,
        excluded_classes=None,
        ignore_index=-1,
        confidence_threshold=0.1,
        vote_k=25,
        enable_voting=True,
        pred_label_mapping=None,
    ):
        """
        Args:
            class_names (list): path to a txt of class names ordered by class index
            text_embeddings (Tensor): path to text embeddings (num_classes, feat_dim)
            excluded_classes (list): Class names to exclude from final metrics
            ignore_index (int): Index to ignore in GT labels
            confidence_threshold (float): Minimum confidence to consider prediction valid
        """
        super().__init__()

        # check if class_names is a list or a string
        if isinstance(class_names, str):
            with open(class_names, "r") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            self.num_classes = len(self.class_names)

            # load text embeddings from the path
            print(f"Loading text embeddings from {text_embeddings}")
            text_embeddings = torch.load(text_embeddings, weights_only=True)
            self.text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            print(
                f"Text embeddings for ZeroShotSemSegEval with shape: {self.text_embeddings.shape}"
            )
            self.excluded_classes = excluded_classes
            self.excluded_indices = [
                i
                for i, name in enumerate(self.class_names)
                if name in (excluded_classes or [])
            ]
            self.ignore_index = ignore_index
            print(f"Excluded classes for ZeroShotSemSegEval: {self.excluded_classes}")
            self.pred_label_mapping = (
                pred_label_mapping  # dict mapping certain pred labels to others
            )
        elif isinstance(class_names, list) or isinstance(class_names, tuple):
            # in this case, we have multiple data to run evaluation on
            # self.class_names, self.text_embeddings, self.excluded_classes, self.excluded_indices will all be list
            (
                self.class_names,
                self.text_embeddings,
                self.excluded_classes,
                self.excluded_indices,
                self.pred_label_mapping,
            ) = [], [], [], [], []
            assert len(class_names) == len(text_embeddings), (
                "class_names and text_embeddings must have the same length"
            )
            for i, class_name_each in enumerate(class_names):
                with open(class_name_each, "r") as f:
                    self.class_names.append(
                        [line.strip() for line in f if line.strip()]
                    )
                self.num_classes = len(self.class_names[-1])

                text_embeddings_each = torch.load(text_embeddings[i], weights_only=True)
                self.text_embeddings.append(
                    F.normalize(text_embeddings_each, p=2, dim=1)
                )
                self.excluded_classes.append(excluded_classes[i])
                self.excluded_indices.append(
                    [
                        j
                        for j, name in enumerate(self.class_names[-1])
                        if name in excluded_classes[i]
                    ]
                )
                self.pred_label_mapping.append(
                    pred_label_mapping[i]
                )  # dict mapping certain pred labels to others

        self.device = torch.device("cuda")
        self.enable_voting = enable_voting
        self.vote_k = vote_k
        self.top_k = 1  # top predictions to consider
        self.confidence_threshold = confidence_threshold
        self.ignore_index = ignore_index

        self.confusion = None
        self.fn_ignore = None
        self._reset_metrics()

    def _reset_metrics(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.fn_ignore = np.zeros(self.num_classes, dtype=np.int64)

    # def after_step(self): # for debug
    #     if self.trainer.cfg.evaluate:
    #         self.eval()
    #         self.trainer.model.train()

    def before_train(self):
        if self.trainer.cfg.enable_wandb:
            safe_wandb_define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def _neighbor_voting(self, coords, initial_labels, valid_mask, query_coords=None):
        """Efficient neighbour voting with optional `query_coords`."""
        if valid_mask is None:
            valid_mask = np.ones(coords.shape[0], dtype=bool)
        valid_coords = coords[valid_mask]
        valid_labels = initial_labels[valid_mask]

        if len(valid_coords) == 0:
            # nothing to vote with
            topk = np.full((len(coords), self.top_k), self.ignore_index, dtype=np.int32)
            return topk

        kd_tree = cKDTree(valid_coords)
        query_pts = coords if query_coords is None else query_coords
        _, nn_idx = kd_tree.query(query_pts, k=self.vote_k)
        if self.vote_k == 1:
            nn_idx = nn_idx[:, None]

        neighbor_labels = valid_labels[nn_idx]

        # fast majority vote for the top‑1 case
        num_classes = getattr(self, "num_classes", int(neighbor_labels.max()) + 1)
        major = _majority_vote(neighbor_labels, self.ignore_index, num_classes)

        # prepare output
        topk_labels = np.full(
            (len(query_pts), self.top_k), self.ignore_index, dtype=np.int32
        )
        topk_labels[:, 0] = major  # always fill the top‑1 slot

        if self.top_k > 1:
            # for loop, only runs when top_k >= 2
            for i in range(len(query_pts)):
                labels = neighbor_labels[i]
                unique, counts = np.unique(labels, return_counts=True)
                if len(unique) == 0:
                    continue
                mask = unique != major[i]
                unique, counts = unique[mask], counts[mask]
                # sort by frequency (desc) then label (asc)
                sort_idx = np.lexsort((unique, -counts))
                extra = unique[sort_idx][: self.top_k - 1]
                topk_labels[i, 1 : 1 + len(extra)] = extra

        del kd_tree, valid_coords, valid_labels, neighbor_labels, nn_idx, major
        if 'query_pts' in locals() and query_pts is not coords:
            del query_pts
        # Force NumPy to release memory pools
        import ctypes
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)  # Force glibc to return memory to OS
        except:
            pass
        gc.collect()

        return topk_labels

    def eval(self):
        self.trainer.model.eval()

        import psutil
        process = psutil.Process()
        print(f"Memory usage before eval: {process.memory_info().rss / 1024**3:.2f} GB")

        # logging
        self.trainer.logger.info(
            ">>>>>>>>>>>>>> Start Zero-Shot SemSeg Evaluation (Multi) >>>>>>>>>>>>>>"
        )
        self.trainer.logger.info(
            f"In total {len(self.class_names)} datasets for evaluation"
        )
        self.trainer.logger.info(
            f"Excluded classes for ZeroShotSemSegEvalMulti: {self.excluded_classes}"
        )
        if self.vote_k > 1 and self.enable_voting:
            self.trainer.logger.info(f"Neighbor voting enabled with k={self.vote_k}")

        len_eval = len(self.class_names)
        all_miou = 0
        for i in range(len_eval):
            text_embeddings = self.text_embeddings[i].to(self.device)
            self.num_classes = len(self.class_names[i])
            self._reset_metrics()
            self.trainer.logger.info(
                f"Evaluating on {i + 1}/{len_eval} val_loader of length {len(self.trainer.val_loader[i])}..."
            )
            print(f"Memory usage before eval: {process.memory_info().rss / 1024**3:.2f} GB")
            with torch.no_grad():
                for j, input_dict in enumerate(self.trainer.val_loader[i]):
                    input_dict = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in input_dict.items()
                    }

                    output_dict = self.trainer.model(input_dict, chunk_size=600000)
                    point_feat = output_dict["point_feat"]["feat"]  # normalized

                    # Get ground truth labels
                    pc_coord = None
                    if "pc_coord" in input_dict and "pc_segment" in input_dict:
                        segment = input_dict.get(
                            "pc_segment",
                            torch.full(
                                (point_feat.size(0),),
                                self.ignore_index,
                                device=point_feat.device,
                            ),
                        )
                        pc_coord = input_dict["pc_coord"].cpu().numpy()
                    else:
                        segment = input_dict.get(
                            "segment",
                            torch.full(
                                (point_feat.size(0),),
                                self.ignore_index,
                                device=point_feat.device,
                            ),
                        )
                    valid_mask = segment != self.ignore_index
                    valid_segment = segment[valid_mask]

                    if valid_segment.numel() == 0:
                        continue

                    logits = torch.mm(point_feat, text_embeddings.t())
                    probs = torch.sigmoid(logits)
                    max_probs, pred_labels = torch.max(probs, dim=1)

                    pred_labels = pred_labels.cpu().numpy()
                    max_probs = max_probs.cpu().numpy()
                    pred_labels[max_probs < self.confidence_threshold] = (
                        self.ignore_index
                    )

                    # Neighbor voting if enabled
                    if self.vote_k > 1 and self.enable_voting:
                        coords = input_dict["coord"].cpu().numpy()
                        valid_feat_mask = input_dict.get("valid_feat_mask", None)
                        topk_labels = self._neighbor_voting(
                            coords=coords,
                            initial_labels=pred_labels,
                            valid_mask=valid_feat_mask.cpu().numpy()
                            if valid_feat_mask is not None
                            else None,
                            query_coords=pc_coord,
                        )
                        pred_labels = topk_labels[:, 0]
                        if "instance" in input_dict:  # clustering voting
                            pred_labels = clustering_voting(
                                pred_labels,
                                input_dict["instance"].cpu().numpy(),
                                self.ignore_index,
                            )

                    # Update confusion matrix
                    valid_pred = pred_labels[valid_mask.cpu().numpy()]
                    valid_gt = valid_segment.cpu().numpy()

                    if self.pred_label_mapping[i]:
                        for key, item in self.pred_label_mapping[i].items():
                            valid_pred[valid_pred == key] = item

                    for gt, pred in zip(valid_gt, valid_pred):
                        if pred == self.ignore_index:
                            self.fn_ignore[gt] += 1
                        else:
                            self.confusion[gt, pred] += 1

                    if (j + 1) % 10 == 0:
                        self.trainer.logger.info(
                            f"Processed {j + 1} / {len(self.trainer.val_loader[i])} batches"
                        )
                    del valid_pred, valid_gt, output_dict, point_feat
                    del input_dict, logits, probs, max_probs, pred_labels
                    gc.collect()
                    # print(f"Memory usage during eval {j + 1} / {len(self.trainer.val_loader[i])}: {process.memory_info().rss / 1024**3:.2f} GB")

            # Synchronize across GPUs in distributed training
            if dist.is_initialized():
                confusion_tensor = torch.tensor(self.confusion).to(self.device)
                fn_ignore_tensor = torch.tensor(self.fn_ignore).to(self.device)
                dist.all_reduce(confusion_tensor)
                dist.all_reduce(fn_ignore_tensor)
                self.confusion = confusion_tensor.cpu().numpy()
                self.fn_ignore = fn_ignore_tensor.cpu().numpy()

            # Compute and log metrics
            fg_miou = self._log_metrics(index=i)
            all_miou += fg_miou
            self.trainer.logger.info(
                f"foreground mIoU: {fg_miou:.4f}, {i + 1}/{len_eval} evaluation with {text_embeddings.shape[0]} classes"
            )

        avg_miou = all_miou / len_eval
        self.trainer.logger.info(
            f"Average f-mIoU: {avg_miou:.4f} for {len_eval} evaluation data"
        )
        self.trainer.comm_info["current_metric_value"] = avg_miou
        self.trainer.comm_info["current_metric_name"] = "avg_fg_mIoU"

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.logger.info(f"Memory usage after eval: {process.memory_info().rss / 1024**3:.2f} GB")
        gc.collect()
        torch.cuda.empty_cache()

    def _log_metrics(self, index=0):
        # ---------- gather IoU for every class ----------
        ious = []
        present_mask = (
            self.confusion.sum(axis=1) + self.fn_ignore
        ) > 0  # no need to store the ignore index in confusion matrix
        present_classes = [c for c in range(self.num_classes) if present_mask[c]]
        included_classes = [
            c for c in present_classes if c not in self.excluded_indices
        ]
        missing_classes = [
            self.class_names[index][c]
            for c in range(self.num_classes)
            if not present_mask[c]
        ]

        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp + self.fn_ignore[c]
            denom = tp + fp + fn
            ious.append(tp / denom if denom > 0 else 0.0)

        # ---------- primary metrics ----------
        metrics = {
            "mIoU": np.mean([ious[c] for c in present_classes]),
            "global_acc": np.diag(self.confusion).sum() / self.confusion.sum(),
            "mean_class_acc": self._calculate_mean_class_acc(present_classes),
            "fg_mIoU": np.mean([ious[c] for c in included_classes])
            if included_classes
            else 0.0,
            "fg_mAcc": self._calculate_mean_class_acc(included_classes)
            if included_classes
            else 0.0,
        }

        # ---------- per-class log ----------
        self.trainer.logger.info("\nMissing classes: %s", missing_classes)
        # self.trainer.logger.info("\n--- Per-class IoU (all classes) ---")
        # for c in present_classes:
        #     self.trainer.logger.info(f"{self.class_names[index][c]:20s}: {ious[c]:.4f}")
        if self.trainer.cfg.enable_wandb:
            for c in present_classes:
                safe_wandb_log(
                    {
                        f"val/data_{index}/cls_{c}-{self.class_names[index][c]} IoU": ious[c],
                    },
                    logger=self.trainer.logger,
                )

        # ---------- main metrics log ----------
        self.trainer.logger.info("\n--- Metrics (ALL present classes) ---")
        self.trainer.logger.info(f"Global Accuracy   : {metrics['global_acc']:.4f}")
        self.trainer.logger.info(f"Mean Class Acc.   : {metrics['mean_class_acc']:.4f}")
        self.trainer.logger.info(f"Mean IoU (mIoU)   : {metrics['mIoU']:.4f}")

        # ----- foreground classes metrics ------
        self.trainer.logger.info(
            f"\n--- Foreground Metrics (EXCLUDED {self.excluded_classes[index]}) ---"
        )
        self.trainer.logger.info(f"Foreground mIoU   : {metrics['fg_mIoU']:.4f}")
        self.trainer.logger.info(f"Foreground mAcc   : {metrics['fg_mAcc']:.4f}")

        # ---------- TensorBoard ----------
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(f"val/data_{index}/mIoU", metrics["mIoU"], current_epoch)
            self.trainer.writer.add_scalar(
                f"val/data_{index}/fg_mIoU", metrics["fg_mIoU"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"val/data_{index}/global_acc", metrics["global_acc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"val/data_{index}/mean_class_acc", metrics["mean_class_acc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"val/data_{index}/fg_mAcc", metrics["fg_mAcc"], current_epoch
            )
            if self.trainer.cfg.enable_wandb:
                safe_wandb_log(
                    {
                        "Epoch": current_epoch,
                        f"val/data_{index}/mIoU": metrics["mIoU"],
                        f"val/data_{index}/fg_mIoU": metrics["fg_mIoU"],
                        f"val/data_{index}/global_acc": metrics["global_acc"],
                        f"val/data_{index}/mAcc": metrics["mean_class_acc"],
                        f"val/data_{index}/fg_mAcc": metrics["fg_mAcc"],
                    },
                    logger=self.trainer.logger,
                )

        return metrics["fg_mIoU"]

    def _calculate_fiou(self, classes, ious):
        """Calculate frequency-weighted IoU for specified classes"""
        total_gt = sum(self.confusion[c].sum() + self.fn_ignore[c] for c in classes)
        if total_gt == 0:
            return 0.0
        return sum(
            (self.confusion[c].sum() + self.fn_ignore[c]) / total_gt * ious[c]
            for c in classes
        )

    def _calculate_mean_class_acc(self, classes):
        """Calculate mean class accuracy for specified classes"""
        accs = []
        for c in classes:
            correct = self.confusion[c, c]
            total = self.confusion[c].sum()
            if total > 0:
                accs.append(correct / total)
        return np.mean(accs) if accs else 0.0

    def _calculate_fw_mean_class_acc(self, classes):
        """frequency-weighted mean class accuracy"""
        total_gt = sum(self.confusion[c].sum() + self.fn_ignore[c] for c in classes)
        if total_gt == 0:
            return 0.0
        fw_acc = 0.0
        for c in classes:
            total = self.confusion[c].sum()
            if total > 0:
                class_acc = self.confusion[c, c] / total
                fw_acc += (total / total_gt) * class_acc
        return fw_acc

    def after_train(self):
        # self.enable_voting = True
        # self.eval()
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )
