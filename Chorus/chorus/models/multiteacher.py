from __future__ import annotations

import copy
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn

from chorus.models.modules import PointSequential
from chorus.models.point_transformer_v3m2_sonata import Block, PointTransformerV3
from chorus.models.utils.structure import Point

DEFAULT_AUTO_MASK_MIN_NORM = 0.01


def _clone_point_for_branch(point: Point) -> Point:
    point_copy = Point(copy.copy(point))
    if "feat" in point_copy:
        point_copy.feat = point_copy.feat.clone()
    if "sparse_conv_feat" in point_copy:
        point_copy.sparse_conv_feat = point_copy.sparse_conv_feat.replace_feature(
            point_copy.feat
        )
    point_copy.pop("pooling_parent", None)
    point_copy.pop("pooling_inverse", None)
    return point_copy


class TeacherProjector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        projector_cfg: dict,
        *,
        clone_inputs: bool = True,
    ) -> None:
        super().__init__()
        cfg = dict(projector_cfg)
        self.clone_inputs = clone_inputs
        block_type = cfg.pop("block_type", "pt-v3m2").lower()
        self.block_type = block_type

        input_norm = cfg.pop("input_norm", True)
        output_norm = cfg.pop("output_norm", False)

        head_module = None
        mlp_module = None
        if block_type == "mlp":
            mlp_ratio = cfg.pop("mlp_ratio", 4.0)
            hidden_channels = max(1, int(round(in_channels * mlp_ratio)))
            self.blocks = None
            mlp_module = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, out_channels),
            )
        elif block_type in {"pt-v3m2", "ptv3m2", "pt-v3m2-sonata"}:
            depth = cfg.pop("depth", 1)
            drop_path_rate = cfg.pop("drop_path_rate", 0.0)
            drop_path_list = cfg.pop("drop_path_list", None)
            if drop_path_list is None:
                drop_path_list = (
                    torch.linspace(0, drop_path_rate, depth).tolist() if depth > 0 else []
                )
            if len(drop_path_list) != depth:
                raise ValueError("drop_path_list length must equal projector depth")

            num_heads = cfg.pop("num_heads", None)
            if num_heads is None:
                raise ValueError("Projector config must specify num_heads")

            patch_size = cfg.pop("patch_size", 1024)
            mlp_ratio = cfg.pop("mlp_ratio", 4.0)
            qkv_bias = cfg.pop("qkv_bias", True)
            qk_scale = cfg.pop("qk_scale", None)
            attn_drop = cfg.pop("attn_drop", 0.0)
            proj_drop = cfg.pop("proj_drop", 0.0)
            layer_scale = cfg.pop("layer_scale", None)
            pre_norm = cfg.pop("pre_norm", True)
            enable_rpe = cfg.pop("enable_rpe", False)
            enable_flash = cfg.pop("enable_flash", True)
            upcast_attention = cfg.pop("upcast_attention", False)
            upcast_softmax = cfg.pop("upcast_softmax", False)

            blocks = []
            for idx in range(depth):
                blocks.append(
                    Block(
                        channels=in_channels,
                        num_heads=num_heads,
                        patch_size=patch_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path_list[idx],
                        layer_scale=layer_scale,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU,
                        pre_norm=pre_norm,
                        order_index=idx,
                        cpe_indice_key=None,
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    )
                )
            self.blocks = PointSequential(*blocks) if blocks else None
            head_module = nn.Linear(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported projector block type: {block_type}")

        self.input_norm = nn.LayerNorm(in_channels) if input_norm else nn.Identity()
        if head_module is not None:
            self.head = head_module
            self.mlp_head = None
        else:
            self.head = None
            self.mlp_head = mlp_module
        self.output_norm = nn.LayerNorm(out_channels) if output_norm else nn.Identity()

    def forward(self, point: Point) -> torch.Tensor:
        point_branch = _clone_point_for_branch(point) if self.clone_inputs else point
        point_branch.feat = self.input_norm(point_branch.feat)
        if self.blocks is not None:
            point_branch = self.blocks(point_branch)
            feat = point_branch.feat
        else:
            feat = point_branch.feat
        feat = self.mlp_head(feat) if self.mlp_head is not None else self.head(feat)
        return self.output_norm(feat)


class LangPretrainerMultiTeacher(nn.Module):
    """Inference-only multi-teacher Chorus encoder.

    The module names intentionally match the training model so released
    checkpoints can be loaded without key rewriting.
    """

    def __init__(
        self,
        backbone: dict,
        teachers: list[dict],
        projector_in_channels: int | None = None,
        training_mode: str = "joint",
        **_: object,
    ) -> None:
        super().__init__()
        if backbone.get("type") not in {"PT-v3m2", "pt-v3m2"}:
            raise ValueError("Detached package mode only supports the PT-v3m2 backbone")
        backbone_cfg = dict(backbone)
        backbone_cfg.pop("type", None)
        self.backbone = PointTransformerV3(**backbone_cfg)
        self.projector_in_channels = projector_in_channels
        self.training_mode = training_mode

        if not teachers:
            raise ValueError("LangPretrainerMultiTeacher requires at least one teacher")

        self.projectors = nn.ModuleDict()
        self.teacher_meta = {}
        self.teacher_order = []
        multi_teacher = len(teachers) > 1

        for raw_teacher_cfg in teachers:
            teacher_cfg = copy.deepcopy(raw_teacher_cfg)
            teacher_name = teacher_cfg.pop("name")
            if teacher_name in self.projectors:
                raise ValueError(f"Duplicated teacher name: {teacher_name}")
            self.teacher_order.append(teacher_name)

            target_key = teacher_cfg.pop("target_key", teacher_name)
            mask_key = teacher_cfg.pop("mask_key", None)
            segment_key = teacher_cfg.pop("segment_key", None)
            loss_weight = teacher_cfg.pop("loss_weight", 1.0)
            mask_min_norm = teacher_cfg.pop("mask_min_norm", DEFAULT_AUTO_MASK_MIN_NORM)
            teacher_cfg.pop("criteria", None)
            teacher_cfg.pop("teacher_norm", None)

            projector_cfg = teacher_cfg.pop("projector", {})
            out_channels = projector_cfg.pop("out_channels", None)
            if out_channels is None:
                raise ValueError(
                    f"Teacher {teacher_name} projector config must include 'out_channels'"
                )
            teacher_in_channels = projector_cfg.pop(
                "in_channels", self.projector_in_channels
            )
            if teacher_in_channels is None:
                raise ValueError(
                    f"Teacher {teacher_name} requires projector input channels"
                )
            clone_inputs = projector_cfg.pop("clone_inputs", multi_teacher)
            self.projectors[teacher_name] = TeacherProjector(
                in_channels=teacher_in_channels,
                out_channels=out_channels,
                projector_cfg=projector_cfg,
                clone_inputs=clone_inputs,
            )
            self.teacher_meta[teacher_name] = dict(
                target_key=target_key,
                mask_key=mask_key,
                segment_key=segment_key,
                loss_weight=loss_weight,
                mask_min_norm=mask_min_norm,
                **teacher_cfg,
            )

        self.register_buffer("_teacher_pointer", torch.zeros(1, dtype=torch.long))

    def _resolve_teacher_names(self, teacher_names: Iterable[str] | None) -> list[str]:
        if teacher_names is None:
            return list(self.teacher_order)
        names = list(dict.fromkeys(teacher_names))
        unknown = [name for name in names if name not in self.projectors]
        if unknown:
            raise KeyError("Unknown teacher names requested: " + ", ".join(unknown))
        return names

    @staticmethod
    def _raw_last_tokens(point: Point) -> dict[str, torch.Tensor]:
        tokens = {"feat": point.feat}
        for key in ("coord", "grid_coord", "offset"):
            if key in point:
                tokens[key] = point[key]
        return tokens

    @staticmethod
    def _upcast_encoder_features(point: Point) -> Point:
        point_feat = point
        while "pooling_parent" in point_feat.keys():
            parent = point_feat.pop("pooling_parent")
            inverse = point_feat.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point_feat.feat[inverse]], dim=-1)
            point_feat = parent
        return point_feat

    def forward(
        self,
        input_dict,
        chunk_size: int | None = None,
        teacher_names: Iterable[str] | None = None,
        return_backbone_upcast: bool = False,
        return_backbone_last: bool = False,
    ) -> dict[str, object]:
        if (
            chunk_size is not None
            and chunk_size > 0
            and input_dict["coord"].shape[0] > chunk_size
        ):
            return self._chunked_forward(
                input_dict,
                chunk_size,
                teacher_names=teacher_names,
                return_backbone_upcast=return_backbone_upcast,
                return_backbone_last=return_backbone_last,
            )

        point = self.backbone(input_dict)
        raw_last = self._raw_last_tokens(point) if return_backbone_last else None
        if isinstance(point, Point) and self.backbone.enc_mode:
            point = self._upcast_encoder_features(point)

        eval_teacher_names = self._resolve_teacher_names(teacher_names)
        projected = {}
        for teacher_name in eval_teacher_names:
            feat = self.projectors[teacher_name](point)
            if self.teacher_meta[teacher_name].get("l2_norm_pred", True):
                feat = nn.functional.normalize(feat, p=2, dim=1)
            projected[teacher_name] = feat

        result: dict[str, object] = {"point_feat": projected}
        if return_backbone_upcast:
            result["backbone_upcast"] = point.feat if isinstance(point, Point) else point
        if raw_last is not None:
            result["backbone_last"] = raw_last
        return result

    def _chunked_forward(
        self,
        input_dict,
        chunk_size: int,
        teacher_names: Iterable[str] | None = None,
        return_backbone_upcast: bool = False,
        return_backbone_last: bool = False,
    ) -> dict[str, object]:
        coords = input_dict["coord"]
        num_points = coords.shape[0]
        device = coords.device
        eval_teacher_names = self._resolve_teacher_names(teacher_names)
        projected_chunks = defaultdict(list)
        upcast_chunks = []
        raw_last_chunks = []

        for start_idx in range(0, num_points, chunk_size):
            end_idx = min(start_idx + chunk_size, num_points)
            chunk_input = {}
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == num_points:
                    chunk_input[key] = value[start_idx:end_idx]
            chunk_input["offset"] = torch.tensor([end_idx - start_idx], device=device)

            out = self.forward(
                chunk_input,
                chunk_size=None,
                teacher_names=eval_teacher_names,
                return_backbone_upcast=return_backbone_upcast,
                return_backbone_last=return_backbone_last,
            )
            for teacher_name, feat in out["point_feat"].items():
                projected_chunks[teacher_name].append(feat)
            if return_backbone_upcast:
                upcast_chunks.append(out["backbone_upcast"])
            if return_backbone_last:
                raw_last_chunks.append(out["backbone_last"])

        projected = {
            teacher_name: torch.cat(chunks, dim=0) if chunks else torch.empty(0, device=device)
            for teacher_name, chunks in projected_chunks.items()
        }
        result: dict[str, object] = {"point_feat": projected}
        if return_backbone_upcast:
            result["backbone_upcast"] = torch.cat(upcast_chunks, dim=0)
        if return_backbone_last:
            result["backbone_last"] = self._concat_raw_last(raw_last_chunks, device)
        return result

    @staticmethod
    def _concat_raw_last(
        chunks: list[dict[str, torch.Tensor]], device: torch.device
    ) -> dict[str, torch.Tensor]:
        if not chunks:
            return {"feat": torch.empty(0, device=device)}
        result = {}
        for key in ("feat", "coord", "grid_coord"):
            values = [chunk[key] for chunk in chunks if key in chunk]
            if values:
                result[key] = torch.cat(values, dim=0)
        offsets = []
        total = 0
        for chunk in chunks:
            count = int(chunk["feat"].shape[0])
            total += count
            offsets.append(total)
        result["offset"] = torch.tensor(offsets, device=device, dtype=torch.long)
        return result
