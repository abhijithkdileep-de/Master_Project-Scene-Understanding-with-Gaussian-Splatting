import logging
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm

import pointcept.utils.comm as comm
from pointcept.models.default import (
    DEFAULT_AUTO_MASK_MIN_NORM,
    TeacherProjector,
    _emit_repo_log,
)
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point

from .builder import MODELS, build_model


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: Optional[int] = None,
    use_bn: bool = False,
    bias: bool = True,
) -> nn.Module:
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)

    if hidden_dim is None:
        hidden_dim = bottleneck_dim

    layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim, eps=1e-3))
    layers.append(nn.GELU())
    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim, eps=1e-3))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)


class MLPHead(nn.Module):
    """Small projector used by the optional 2D adaptation light-projector path."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
        normalize: bool = True,
        remove_last_layer: bool = False,
    ) -> None:
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.apply(self._init_weights)
        self.remove_last_layer = remove_last_layer
        if not self.remove_last_layer:
            self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
        self.normalize = normalize

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-4)
        if not self.remove_last_layer:
            x = self.last_layer(x)
        return x


def _lazy_build_teacher_2d(
    model_name: str,
) -> Tuple[nn.Module, Callable, str]:
    """Build a 2D teacher only when the 2D adaptation model is instantiated."""

    model_name_lower = model_name.lower()
    if "dinov3" in model_name_lower:
        from pointcept.models.feature_extract_2D.dinov3_model import (
            DINOv3_Wrapper,
            image_from_path,
        )

        return DINOv3_Wrapper(model_name=model_name), image_from_path, "dinov3"

    if "siglip2" in model_name_lower:
        from pointcept.models.feature_extract_2D.siglip2_model import (
            Siglip2_Wrapper,
            image_from_path_siglip,
        )

        return Siglip2_Wrapper(model_name=model_name), image_from_path_siglip, "siglip2"

    if "pe-spatial" in model_name_lower or "pe_spatial" in model_name_lower:
        from pointcept.models.feature_extract_2D.pe_spatial_model import (
            PE_Spatial_Wrapper,
            image_from_path_pe_spatial,
        )

        return (
            PE_Spatial_Wrapper(model_name=model_name),
            image_from_path_pe_spatial,
            "pe_spatial",
        )

    raise ValueError(f"Unsupported 2D teacher model: {model_name}")


def _lazy_peft_lora():
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "LoRA-based 2D adaptation requires peft. Install the dependencies from env.yaml."
        ) from exc
    return LoraConfig, get_peft_model


def _extract_backbone_state(
    checkpoint: Dict,
    *,
    keywords: Optional[str] = None,
    replacements: Optional[str] = None,
) -> OrderedDict:
    state_dict = checkpoint.get("state_dict", checkpoint)
    weight = OrderedDict()
    for key, value in state_dict.items():
        if not key.startswith("module."):
            key = "module." + key
        if keywords is not None and replacements is not None and keywords in key:
            key = key.replace(keywords, replacements)
        key = key[7:]
        if key.startswith("backbone."):
            key = key[9:]
        weight[key] = value
    return weight


def _set_lora_trainable(backbone: nn.Module, freeze_backbone: bool, use_lora: bool) -> None:
    if freeze_backbone:
        for name, param in backbone.named_parameters():
            param.requires_grad = use_lora and "lora_" in name
        return

    if use_lora:
        for name, param in backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True


@MODELS.register_module()
class LangPretrainerMultiTeacher2D(nn.Module):
    """Optional 2D-adaptation trainer for Chorus.

    This class intentionally keeps all gsplat and 2D teacher imports lazy so the
    normal LangPretrainerMultiTeacher workflow can import without 2D extras.
    """

    def __init__(
        self,
        backbone=None,
        teachers=None,
        projector_in_channels=None,
        training_mode="joint",
        resize_w=640,
        resize_h=480,
        image_upsample_factor=1,
        use_lora=False,
        backbone_path=None,
        keywords=None,
        replacements=None,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        freeze_backbone=False,
        freeze_projector=False,
        online_image=True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            _emit_repo_log(
                logging.INFO,
                "Unused 2D adaptation model config keys: %s",
                list(kwargs.keys()),
            )
        self.backbone = build_model(backbone)
        self.freeze_backbone = freeze_backbone
        self.freeze_projector = freeze_projector
        self.image_upsample_factor = image_upsample_factor
        self.use_lora = use_lora
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.online_image = online_image
        self.projector_in_channels = projector_in_channels
        self.keywords = keywords
        self.replacements = replacements

        if backbone_path is not None:
            checkpoint = torch.load(backbone_path, map_location="cpu")
            self.backbone_load(checkpoint)

        if self.use_lora:
            LoraConfig, get_peft_model = _lazy_peft_lora()
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)
        _set_lora_trainable(self.backbone, self.freeze_backbone, self.use_lora)

        if teachers is None or len(teachers) == 0:
            raise ValueError("LangPretrainerMultiTeacher2D requires at least one teacher")
        self.training_mode = training_mode
        if self.training_mode not in {"joint", "alternating"}:
            raise ValueError("training mode must be one of {'joint', 'alternating'}")

        self.projectors = nn.ModuleDict()
        self.teacher_norms = nn.ModuleDict()
        self.teacher_model_2D = nn.ModuleDict()
        self.criteria = {}
        self.teacher_meta = {}
        self.teacher_model_2D_meta = {}
        self.teacher_order = []

        multi_teacher = len(teachers) > 1
        for teacher_cfg in teachers:
            teacher_cfg = teacher_cfg.copy()
            teacher_name = teacher_cfg.get("name")
            if teacher_name is None:
                raise ValueError("Each teacher must define a unique 'name'")
            if teacher_name in self.projectors:
                raise ValueError(f"Duplicated teacher name: {teacher_name}")
            self.teacher_order.append(teacher_name)

            teacher_2d_model_name = teacher_cfg.get("teacher_2D_model")
            if teacher_2d_model_name is None:
                raise ValueError(
                    f"Teacher {teacher_name} must define 'teacher_2D_model' for 2D adaptation"
                )
            teacher_2d_model, image_loader, teacher_kind = _lazy_build_teacher_2d(
                teacher_2d_model_name
            )
            self.teacher_model_2D[teacher_name] = teacher_2d_model

            light_projector = bool(teacher_cfg.get("light_projector", False))
            downsample_ratio_2d = teacher_cfg.get("downsample_ratio_2D", 4)
            patch_size_2d = getattr(teacher_2d_model, "patch_size", 16)
            self.teacher_model_2D_meta[teacher_name] = dict(
                downsample_ratio_2D=downsample_ratio_2d,
                patch_size_2D=patch_size_2d,
                light_projector=light_projector,
                image_loader=image_loader,
                teacher_kind=teacher_kind,
            )

            projector_cfg = teacher_cfg.get("projector", {}).copy()
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
                    f"Teacher {teacher_name} projector requires 'in_channels' or a global 'projector_in_channels'"
                )
            clone_inputs = projector_cfg.pop("clone_inputs", None)
            if clone_inputs is None:
                clone_inputs = multi_teacher
            elif not isinstance(clone_inputs, bool):
                raise TypeError(
                    f"Teacher {teacher_name} projector 'clone_inputs' must be a bool when provided"
                )

            if light_projector:
                projector = MLPHead(
                    in_dim=teacher_in_channels,
                    out_dim=out_channels,
                    hidden_dim=2048,
                    bottleneck_dim=out_channels,
                    nlayers=3,
                    normalize=True,
                    remove_last_layer=True,
                )
            else:
                projector = TeacherProjector(
                    in_channels=teacher_in_channels,
                    out_channels=out_channels,
                    projector_cfg=projector_cfg,
                    clone_inputs=clone_inputs,
                )
            self.projectors[teacher_name] = projector
            self.teacher_norms[teacher_name] = nn.Identity()
            self.criteria[teacher_name] = build_criteria(teacher_cfg.get("criteria", []))
            self.teacher_meta[teacher_name] = dict(
                target_key=teacher_cfg.get("target_key", teacher_name),
                mask_key=teacher_cfg.get("mask_key", None),
                segment_key=teacher_cfg.get("segment_key", None),
                loss_weight=teacher_cfg.get("loss_weight", 1.0),
                mask_min_norm=teacher_cfg.get(
                    "mask_min_norm", DEFAULT_AUTO_MASK_MIN_NORM
                ),
            )

        if self.freeze_projector:
            for projector in self.projectors.values():
                for param in projector.parameters():
                    param.requires_grad = False

        self.register_buffer("_teacher_pointer", torch.zeros(1, dtype=torch.long))
        if comm.is_main_process():
            self.count_params()

    def backbone_load(self, checkpoint):
        weight = _extract_backbone_state(
            checkpoint, keywords=self.keywords, replacements=self.replacements
        )
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        _emit_repo_log(
            logging.INFO,
            "[2DAdapt] Loaded backbone. Missing keys: %d, unexpected keys: %d",
            len(load_state_info.missing_keys),
            len(load_state_info.unexpected_keys),
        )

    def count_params(self):
        def _count_parameters(module: nn.Module) -> int:
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        backbone_total = _count_parameters(getattr(self.backbone, "enc", None))
        backbone_total += _count_parameters(getattr(self.backbone, "dec", None))
        backbone_total += _count_parameters(getattr(self.backbone, "embedding", None))
        _emit_repo_log(
            logging.INFO,
            "[ParamCount][2DAdaptBackbone] Trainable: %s",
            format(backbone_total, ","),
        )
        for name, projector in self.projectors.items():
            _emit_repo_log(
                logging.INFO,
                "[ParamCount][2DAdaptProjector:%s] Trainable: %s",
                name,
                format(_count_parameters(projector), ","),
            )

    def _project_teacher(self, teacher_name: str, point_feat: Point) -> torch.Tensor:
        projector = self.projectors[teacher_name]
        if self.teacher_model_2D_meta[teacher_name]["light_projector"]:
            feat = point_feat.feat if isinstance(point_feat, Point) else point_feat
            return projector(feat)
        return projector(point_feat)

    def _load_offline_image(self, teacher_name: str, image_path: str):
        loader = self.teacher_model_2D_meta[teacher_name]["image_loader"]
        return loader(image_path, normalize=True, resize_down=None, crop_edge=0.0)

    @staticmethod
    def _normalize_online_image(img: torch.Tensor, teacher_kind: str) -> torch.Tensor:
        img = img.permute(2, 0, 1).contiguous()
        if teacher_kind == "dinov3":
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
            return (img - mean) / std
        if teacher_kind == "pe_spatial":
            mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(3, 1, 1)
            return (img - mean) / std
        return img

    def _teacher_feature_map(
        self,
        teacher_name: str,
        teacher_model_2d: nn.Module,
        image,
        downsample_ratio_2d: int,
    ) -> torch.Tensor:
        feat_map = teacher_model_2d.predict(image).unsqueeze(0)
        feat_map = nn.functional.interpolate(
            feat_map,
            size=(
                self.resize_h // downsample_ratio_2d,
                self.resize_w // downsample_ratio_2d,
            ),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return feat_map.permute(1, 2, 0).contiguous()

    def _compute_teacher_loss_2D(
        self,
        teacher_name,
        point_feat,
        input_dict,
        gs_dict,
        ks_batch,
        viewmats_batch,
        image_paths_batch,
        batch_offset,
        colors_list=None,
    ):
        from pointcept.utils.gsplat_utils import (
            rasterize_multiple_gaussians_to_multiple_feats,
        )

        meta = self.teacher_meta[teacher_name]
        teacher_model_2d = self.teacher_model_2D[teacher_name]
        teacher_2d_meta = self.teacher_model_2D_meta[teacher_name]
        downsample_ratio_2d = teacher_2d_meta["downsample_ratio_2D"]
        teacher_kind = teacher_2d_meta["teacher_kind"]

        pred = self._project_teacher(teacher_name, point_feat)
        pred = nn.functional.normalize(pred, p=2, dim=1)

        pred_list = []
        offset = torch.cat((torch.tensor([0], device=batch_offset.device), batch_offset))
        for batch_idx in range(len(offset) - 1):
            start, end = offset[batch_idx], offset[batch_idx + 1]
            pred_list.append(pred[start:end].float())

        feat_list_from_teacher = []
        if self.online_image:
            if colors_list is None:
                raise RuntimeError("online_image=True requires rendered color images")
            count = 0
            for image_batch in image_paths_batch:
                for _ in image_batch:
                    img = colors_list[count][0]
                    count += 1
                    img = self._normalize_online_image(img, teacher_kind)
                    feat_list_from_teacher.append(
                        self._teacher_feature_map(
                            teacher_name,
                            teacher_model_2d,
                            img,
                            downsample_ratio_2d,
                        )
                    )
        else:
            for image_batch in image_paths_batch:
                for image_path in image_batch:
                    img = self._load_offline_image(teacher_name, image_path)
                    if isinstance(img, torch.Tensor):
                        _, height, width = img.shape
                        patch_size = teacher_2d_meta["patch_size_2D"]
                        height_aligned = (height // patch_size) * patch_size
                        width_aligned = (width // patch_size) * patch_size
                        if self.image_upsample_factor != 1:
                            height_aligned *= self.image_upsample_factor
                            width_aligned *= self.image_upsample_factor
                        if height_aligned != height or width_aligned != width:
                            img = nn.functional.interpolate(
                                img[None],
                                size=(height_aligned, width_aligned),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0)
                    feat_list_from_teacher.append(
                        self._teacher_feature_map(
                            teacher_name,
                            teacher_model_2d,
                            img,
                            downsample_ratio_2d,
                        )
                    )

        feat_list_from_teacher = torch.stack(feat_list_from_teacher, dim=0)
        predicted_feats, valid_feats = rasterize_multiple_gaussians_to_multiple_feats(
            gs_dict,
            viewmats_batch,
            ks_batch,
            self.resize_w,
            self.resize_h,
            downsample_ratio=downsample_ratio_2d,
            features_3d_list=pred_list,
            image_paths_batch=image_paths_batch,
            save_visualize=False,
            need_grad=True,
        )

        feat_list_from_teacher = nn.functional.normalize(
            feat_list_from_teacher, p=2, dim=-1, eps=1e-6
        )
        predicted_feats = nn.functional.normalize(
            predicted_feats, p=2, dim=-1, eps=1e-6
        )

        pred_flat = predicted_feats.view(-1, predicted_feats.shape[-1])
        target_flat = feat_list_from_teacher.view(-1, feat_list_from_teacher.shape[-1])
        mask = valid_feats.view(-1).bool()
        loss = self.criteria[teacher_name](
            pred_flat, target_flat, valid_feat_mask=mask
        )
        return dict(
            loss=loss * meta.get("loss_weight", 1.0),
            raw_loss=loss,
            pred=pred_flat,
            mask=mask,
        )

    def _advance_pointer(self):
        self._teacher_pointer.add_(1)
        if self._teacher_pointer.item() >= len(self.teacher_order):
            self._teacher_pointer.zero_()

    @staticmethod
    def _parse_image_paths(raw_image_paths):
        if isinstance(raw_image_paths, str):
            return [raw_image_paths.split(";")]
        image_paths_batch = []
        for image_paths in raw_image_paths:
            if isinstance(image_paths, str):
                image_paths_batch.append(image_paths.split(";"))
            else:
                image_paths_batch.append(list(image_paths))
        return image_paths_batch

    def forward(self, input_dict, chunk_size=None):
        del chunk_size
        from pointcept.utils.gsplat_utils import (
            build_gs_dict,
            rasterize_multiple_gaussians_to_multiple_imgs,
        )

        ks_batch = input_dict["K"]
        viewmats_batch = input_dict["poses"]
        batch_offset = input_dict["offset"]
        image_paths_batch = self._parse_image_paths(input_dict["image_paths"])

        gs_dict = build_gs_dict(
            input_dict["coord"].detach(),
            input_dict["feat"].detach(),
            input_dict["offset"],
        )

        colors_list = None
        if self.online_image:
            colors_list = rasterize_multiple_gaussians_to_multiple_imgs(
                gs_dict,
                viewmats_batch,
                ks_batch,
                self.resize_w,
                self.resize_h,
                image_paths_batch,
                save_visualize=False,
            )

        point = Point(input_dict)
        point_feat = self.backbone(point)
        if isinstance(point_feat, Point) and self.backbone.enc_mode:
            while "pooling_parent" in point_feat.keys():
                assert "pooling_inverse" in point_feat.keys()
                parent = point_feat.pop("pooling_parent")
                inverse = point_feat.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_feat.feat[inverse]], dim=-1)
                point_feat = parent

        if self.training:
            if self.training_mode == "alternating":
                teacher_name = self.teacher_order[int(self._teacher_pointer.item())]
                result = self._compute_teacher_loss_2D(
                    teacher_name,
                    point_feat,
                    input_dict,
                    gs_dict,
                    ks_batch,
                    viewmats_batch,
                    image_paths_batch,
                    batch_offset,
                    colors_list,
                )
                self._advance_pointer()
                return dict(
                    loss=result["loss"],
                    active_teacher=teacher_name,
                    per_teacher_loss={teacher_name: result["raw_loss"].detach()},
                )

            total_loss = 0.0
            per_teacher_loss = {}
            for teacher_name in self.teacher_order:
                result = self._compute_teacher_loss_2D(
                    teacher_name,
                    point_feat,
                    input_dict,
                    gs_dict,
                    ks_batch,
                    viewmats_batch,
                    image_paths_batch,
                    batch_offset,
                    colors_list,
                )
                total_loss = total_loss + result["loss"]
                per_teacher_loss[teacher_name] = result["raw_loss"].detach()
            return dict(loss=total_loss, per_teacher_loss=per_teacher_loss)

        projected = {}
        for teacher_name in self.teacher_order:
            feat = self._project_teacher(teacher_name, point_feat)
            projected[teacher_name] = nn.functional.normalize(feat, p=2, dim=1)
        return dict(point_feat=projected)


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        if backbone_path is not None:
            checkpoint = torch.load(backbone_path, map_location="cpu")
            self.backbone_load(checkpoint)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            LoraConfig, get_peft_model = _lazy_peft_lora()
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)
        _set_lora_trainable(self.backbone, self.freeze_backbone, self.use_lora)

    def backbone_load(self, checkpoint):
        weight = _extract_backbone_state(
            checkpoint, keywords=self.keywords, replacements=self.replacements
        )
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        _emit_repo_log(
            logging.INFO,
            "[DefaultLORASegmentorV2] Loaded backbone. Missing keys: %d, unexpected keys: %d",
            len(load_state_info.missing_keys),
            len(load_state_info.unexpected_keys),
        )

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            return_dict["loss"] = self.criteria(seg_logits, input_dict["segment"])
        elif "segment" in input_dict:
            return_dict["loss"] = self.criteria(seg_logits, input_dict["segment"])
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict
