import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.distributed as dist
import torch.nn.functional as F
import gc

from uuid import uuid4
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pointcept.utils.misc import (
    intersection_and_union_gpu,
    clustering_voting,
    _majority_vote,
    inspect_memory,
)

@dataclass
class ZeroShotEvalResult:
    metrics: Dict[str, float]
    ious: List[float]
    present_classes: List[int]
    included_classes: List[int]
    missing_classes: List[str]


class ZeroShotSemSegEvaluatorHelper:
    """Reusable core for zero-shot semantic segmentation evaluation."""

    def __init__(
        self,
        class_names: List[str],
        text_embeddings: torch.Tensor,
        excluded_classes: Optional[List[str]] = None,
        ignore_index: int = -1,
        confidence_threshold: float = 0.1,
        vote_k: int = 25,
        enable_voting: bool = True,
        pred_label_mapping: Optional[Dict[int, int]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_embeddings = F.normalize(text_embeddings, p=2, dim=1).to(self.device)
        self.excluded_classes = excluded_classes or []
        self.excluded_indices = [
            idx for idx, name in enumerate(class_names) if name in self.excluded_classes or 'other' in name.lower()
        ]
        self.ignore_index = ignore_index
        self.confidence_threshold = confidence_threshold
        self.vote_k = vote_k
        self.enable_voting = enable_voting
        self.top_k = 1
        self.pred_label_mapping = pred_label_mapping or {}

        self.reset()

    def reset(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.fn_ignore = np.zeros(self.num_classes, dtype=np.int64)

    def _neighbor_voting(
        self,
        coords: np.ndarray,
        initial_labels: np.ndarray,
        valid_mask: Optional[np.ndarray],
        query_coords: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if valid_mask is None:
            valid_mask = np.ones(coords.shape[0], dtype=bool)
        valid_coords = coords[valid_mask]
        valid_labels = initial_labels[valid_mask]

        if len(valid_coords) == 0:
            query_len = len(coords if query_coords is None else query_coords)
            return np.full((query_len, self.top_k), self.ignore_index, dtype=np.int32)

        kd_tree = cKDTree(valid_coords)
        query_pts = coords if query_coords is None else query_coords
        _, nn_idx = kd_tree.query(query_pts, k=self.vote_k)
        if self.vote_k == 1:
            nn_idx = nn_idx[:, None]
        neighbor_labels = valid_labels[nn_idx]

        major = _majority_vote(neighbor_labels, self.ignore_index, self.num_classes)
        topk_labels = np.full((len(query_pts), self.top_k), self.ignore_index, dtype=np.int32)
        topk_labels[:, 0] = major

        if self.top_k > 1:
            for idx in range(len(query_pts)):
                labels = neighbor_labels[idx]
                unique, counts = np.unique(labels, return_counts=True)
                if len(unique) == 0:
                    continue
                mask = unique != major[idx]
                unique, counts = unique[mask], counts[mask]
                sort_idx = np.lexsort((unique, -counts))
                extra = unique[sort_idx][: self.top_k - 1]
                topk_labels[idx, 1 : 1 + len(extra)] = extra

        del kd_tree, neighbor_labels, nn_idx
        if query_coords is not None and query_coords is not coords:
            del query_coords
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        gc.collect()
        return topk_labels

    def process_batch(self, input_dict: Dict, point_feat: torch.Tensor) -> None:
        segment = None
        pc_coord = None
        project_to_pc = (
            self.enable_voting
            and "pc_coord" in input_dict
            and "pc_segment" in input_dict
        )
        if project_to_pc:
            segment = input_dict["pc_segment"]
            pc_coord = input_dict["pc_coord"].cpu().numpy()
        else:
            segment = input_dict.get(
                "segment",
                torch.full((point_feat.size(0),), self.ignore_index, device=point_feat.device),
            )

        valid_mask = segment != self.ignore_index
        if valid_mask.sum().item() == 0:
            return

        text_embeddings = self.text_embeddings
        logits = torch.mm(point_feat, text_embeddings.t())
        probs = torch.sigmoid(logits)
        max_probs, pred_labels = torch.max(probs, dim=1)

        pred_labels = pred_labels.detach().cpu().numpy()
        max_probs = max_probs.detach().cpu().numpy()
        pred_labels[max_probs < self.confidence_threshold] = self.ignore_index

        if self.enable_voting:
            coords = input_dict["coord"].cpu().numpy()
            valid_feat_mask = input_dict.get("valid_feat_mask", None)
            if isinstance(valid_feat_mask, torch.Tensor):
                valid_feat_mask = valid_feat_mask.cpu().numpy().astype(bool)
            elif valid_feat_mask is not None:
                valid_feat_mask = np.asarray(valid_feat_mask).astype(bool)
            topk_labels = self._neighbor_voting(
                coords=coords,
                initial_labels=pred_labels,
                valid_mask=valid_feat_mask,
                query_coords=pc_coord,
            )
            pred_labels = topk_labels[:, 0]
            instance = input_dict.get("pc_instance" if project_to_pc else "instance", None)
            if isinstance(instance, torch.Tensor):
                instance = instance.cpu().numpy()
            elif instance is not None:
                instance = np.asarray(instance)
            if instance is not None and len(instance) == len(pred_labels):
                pred_labels = clustering_voting(
                    pred_labels,
                    instance,
                    self.ignore_index,
                )

        valid_pred = pred_labels[valid_mask.cpu().numpy()]
        valid_gt = segment[valid_mask].cpu().numpy()

        if self.pred_label_mapping:
            for key, item in self.pred_label_mapping.items():
                valid_pred[valid_pred == key] = item

        for gt, pred in zip(valid_gt, valid_pred):
            if pred == self.ignore_index:
                self.fn_ignore[gt] += 1
            else:
                self.confusion[gt, pred] += 1

        del logits, probs, max_probs, pred_labels, valid_pred, valid_gt
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        if dist.is_initialized():
            confusion_tensor = torch.tensor(self.confusion, device=self.device)
            fn_ignore_tensor = torch.tensor(self.fn_ignore, device=self.device)
            dist.all_reduce(confusion_tensor)
            dist.all_reduce(fn_ignore_tensor)
            self.confusion = confusion_tensor.cpu().numpy()
            self.fn_ignore = fn_ignore_tensor.cpu().numpy()

    def _calculate_mean_class_acc(self, classes: List[int]) -> float:
        accs = []
        for c in classes:
            total = self.confusion[c].sum()
            if total > 0:
                accs.append(self.confusion[c, c] / total)
        return float(np.mean(accs)) if accs else 0.0

    def compute_metrics(self) -> ZeroShotEvalResult:
        present_mask = (self.confusion.sum(axis=1) + self.fn_ignore) > 0
        present_classes = [idx for idx in range(self.num_classes) if present_mask[idx]]
        included_classes = [idx for idx in present_classes if idx not in self.excluded_indices]
        missing_classes = [
            self.class_names[idx] for idx in range(self.num_classes) if not present_mask[idx]
        ]

        ious = []
        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp + self.fn_ignore[c]
            denom = tp + fp + fn
            ious.append(tp / denom if denom > 0 else 0.0)

        metrics = {
            "mIoU": np.mean([ious[c] for c in present_classes]) if present_classes else 0.0,
            "global_acc": np.diag(self.confusion).sum() / self.confusion.sum()
            if self.confusion.sum() > 0
            else 0.0,
            "mean_class_acc": self._calculate_mean_class_acc(present_classes)
            if present_classes
            else 0.0,
            "fg_mIoU": np.mean([ious[c] for c in included_classes])
            if included_classes
            else 0.0,
            "fg_mAcc": self._calculate_mean_class_acc(included_classes)
            if included_classes
            else 0.0,
        }

        return ZeroShotEvalResult(
            metrics=metrics,
            ious=ious,
            present_classes=present_classes,
            included_classes=included_classes,
            missing_classes=missing_classes,
        )


class FeatureSimilarityEvaluatorHelper:
    """Compute cosine similarity and L2 metrics between predictions and teacher targets."""

    def __init__(
        self,
        target_key: str,
        mask_key: Optional[str] = None,
        mask_min_norm: float = 0.05,
        sample_stride: int = 4,
        chunk_size: int = 200_000,
    ) -> None:
        self.target_key = target_key
        self.mask_key = mask_key
        self.mask_min_norm = mask_min_norm
        self.sample_stride = max(1, int(sample_stride))
        self.chunk_size = max(1, int(chunk_size))
        self.reset()

    def reset(self) -> None:
        self.sum_cosine = 0.0
        self.sum_l2 = 0.0
        self.count = 0

    def __str__(self):
        return (
            f"FeatureSimilarityEvaluatorHelper(target_key={self.target_key}, "
            f"mask_key={self.mask_key}, mask_min_norm={self.mask_min_norm}, "
            f"sample_stride={self.sample_stride}, chunk_size={self.chunk_size})"
        )

    def _infer_mask(self, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            return mask.bool()
        norms = target.norm(dim=1)
        return norms > float(self.mask_min_norm)

    def process_batch(self, input_dict: Dict, pred_feat: torch.Tensor) -> None:
        device = pred_feat.device
        target = input_dict[self.target_key]
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, device=device, dtype=pred_feat.dtype)
        else:
            target = target.to(device=device, dtype=pred_feat.dtype)

        mask_tensor = None
        if self.mask_key and self.mask_key in input_dict:
            mask_tensor = input_dict[self.mask_key]
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.as_tensor(mask_tensor, device=device)
            else:
                mask_tensor = mask_tensor.to(device=device)

        mask = self._infer_mask(target, mask_tensor)
        valid_mask = mask.bool()
        if valid_mask.numel() == 0 or valid_mask.sum().item() == 0:
            return

        indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            return

        if self.sample_stride > 1:
            indices = indices[:: self.sample_stride]
            if indices.numel() == 0:
                return

        for start in range(0, indices.numel(), self.chunk_size):
            end = min(start + self.chunk_size, indices.numel())
            idx_chunk = indices[start:end]
            pred_chunk = pred_feat.index_select(0, idx_chunk)
            target_chunk = target.index_select(0, idx_chunk)

            cos_values = F.cosine_similarity(pred_chunk, target_chunk, dim=1)
            l2_values = (pred_chunk - target_chunk).pow(2).sum(dim=1)

            self.sum_cosine += cos_values.sum().item()
            self.sum_l2 += l2_values.sum().item()
            self.count += cos_values.numel()

            del pred_chunk, target_chunk, cos_values, l2_values
            torch.cuda.empty_cache()

    def compute_metrics(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "cosine_similarity": 0.0,
                "cosine_loss": 1.0,
                "l2_loss": 0.0,
            }
        mean_cos = self.sum_cosine / self.count
        mean_l2 = self.sum_l2 / self.count
        return {
            "cosine_similarity": mean_cos,
            "cosine_loss": 1.0 - mean_cos,
            "l2_loss": mean_l2,
        }
