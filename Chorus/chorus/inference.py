from __future__ import annotations

import json
import logging
import os
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from chorus.checkpoints import load_checkpoint, resolve_checkpoint_reference
from chorus.collate import collate_fn
from chorus.configs import SUPPORTED_OUTPUTS, TEACHER_OUTPUTS, get_preset
from chorus.input import GaussianReadResult, load_gaussian_input
from chorus.models import LangPretrainerMultiTeacher
from chorus.transforms import Compose, build_transform


@dataclass
class TokenOutput:
    feat: Any
    coord: Any | None = None
    grid_coord: Any | None = None
    offset: Any | None = None
    fragment_index: Any | None = None


@dataclass
class ChorusOutput:
    name: str
    features: dict[str, Any]
    tokens: dict[str, TokenOutput]
    metadata: dict[str, Any]

    @property
    def backbone_last(self) -> TokenOutput | None:
        return self.tokens.get("backbone_last")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("chorus.inference")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def _normalize_outputs(outputs) -> tuple[str, ...]:
    if isinstance(outputs, str):
        outputs = [item.strip() for item in outputs.split(",") if item.strip()]
    outputs = tuple(outputs)
    unknown = sorted(set(outputs) - SUPPORTED_OUTPUTS)
    if unknown:
        raise ValueError(
            "Unsupported Chorus output(s): "
            + ", ".join(unknown)
            + f". Supported outputs: {', '.join(sorted(SUPPORTED_OUTPUTS))}"
        )
    if not outputs:
        raise ValueError("At least one output must be requested")
    return outputs


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


class ChorusEncoder:
    def __init__(
        self,
        *,
        mode: str = "chorus_3dgs",
        checkpoint: str | None = None,
        outputs=("lang",),
        device: str = "cuda",
        return_numpy: bool = True,
        chunk_size: int | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        outlier_filter: bool | Mapping[str, Any] | None = None,
        checkpoint_hub: Mapping[str, Any] | None = None,
    ) -> None:
        self.cfg = get_preset(mode)
        self.mode = mode
        self.outputs = _normalize_outputs(outputs)
        self.return_numpy = return_numpy
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        self.logger = _setup_logger()

        requested_device = device or "cuda"
        if not str(requested_device).startswith("cuda"):
            raise RuntimeError(
                "Chorus package-mode encoding currently requires CUDA. "
                "CPU execution is not supported by the PT-v3m2 + spconv runtime."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Chorus package-mode encoding requires CUDA, but no CUDA device is available."
            )
        self.device = torch.device(requested_device)

        if chunk_size is not None:
            self.cfg["chunk_size"] = chunk_size
        if checkpoint_hub is not None:
            self.cfg["checkpoint_hub"] = dict(checkpoint_hub)
        if outlier_filter is not None:
            self._override_outlier_filter(outlier_filter)

        self.feat_keys = tuple(self.cfg["feat_keys"])
        self.transform = Compose(self.cfg.get("transform", []))
        test_cfg = self.cfg.get("test_cfg", {})
        self.test_voxelize = (
            build_transform(test_cfg["voxelize"])
            if test_cfg and test_cfg.get("voxelize")
            else None
        )
        self.test_crop = (
            build_transform(test_cfg["crop"]) if test_cfg and test_cfg.get("crop") else None
        )
        self.post_transform = Compose(test_cfg.get("post_transform", []) if test_cfg else [])
        self.aug_transform = [
            Compose(cfg_) for cfg_ in (test_cfg.get("aug_transform", [[]]) if test_cfg else [[]])
        ]
        self.chunk_size = self.cfg.get("chunk_size")
        self.default_scene_name = self.cfg.get("default_scene_name", "inference_sample")

        checkpoint_ref = checkpoint or self.cfg["checkpoint"]
        self.checkpoint_info = resolve_checkpoint_reference(
            checkpoint_ref,
            hub_cfg=self.cfg.get("checkpoint_hub"),
            logger=self.logger,
        )
        model_cfg = dict(self.cfg["model"])
        model_cfg.pop("type", None)
        self.model = LangPretrainerMultiTeacher(**model_cfg)
        self.model.eval()
        self.model.to(self.device)
        load_info = load_checkpoint(self.model, self.checkpoint_info["local_path"], strict=False)
        missing = list(load_info["load_state_info"].missing_keys)
        unexpected = list(load_info["load_state_info"].unexpected_keys)
        if missing:
            self.logger.warning("Missing checkpoint keys: %s", missing[:20])
        if unexpected:
            self.logger.warning("Unexpected checkpoint keys: %s", unexpected[:20])

        unknown_teachers = sorted((set(self.outputs) & TEACHER_OUTPUTS) - set(self.model.teacher_order))
        if unknown_teachers:
            raise KeyError("Requested unknown teacher outputs: " + ", ".join(unknown_teachers))

    def _override_outlier_filter(self, outlier_filter: bool | Mapping[str, Any]) -> None:
        reader_cfg = self.cfg.setdefault("input_reader", {})
        current = reader_cfg.setdefault("outlier_filter", {})
        if isinstance(outlier_filter, bool):
            current["enabled"] = outlier_filter
        else:
            current.update(dict(outlier_filter))

    @property
    def active_teachers(self) -> list[str]:
        return [name for name in self.model.teacher_order if name in self.outputs]

    def describe_runtime(self) -> dict[str, Any]:
        return dict(
            mode=self.mode,
            outputs=list(self.outputs),
            checkpoint=self.checkpoint_info,
            chunk_size=self.chunk_size,
            output_dir=str(self.output_dir) if self.output_dir else None,
            return_numpy=self.return_numpy,
            test_cfg=_to_builtin(self.cfg.get("test_cfg", {})),
            input_reader=_to_builtin(self.cfg.get("input_reader", {})),
        )

    def encode(
        self,
        input_root: str | os.PathLike[str] | Mapping[str, np.ndarray],
        *,
        scene_name: str | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        save: bool | None = None,
        outlier_filter: bool | Mapping[str, Any] | None = None,
    ) -> ChorusOutput:
        read_result = self._read_input(input_root, scene_name, outlier_filter)
        target_output_dir = Path(output_dir).expanduser() if output_dir else self.output_dir
        should_save = bool(target_output_dir) if save is None else save
        output = self._encode_arrays(
            read_result.data,
            scene_name=read_result.scene_name,
            metadata=dict(
                input_path=read_result.input_path,
                source_type=read_result.source_type,
                source_raw_count=read_result.raw_count,
                source_kept_count=read_result.kept_count,
                source_keep_index=read_result.kept_indices,
                outlier_filter_enabled=read_result.outlier_filter_enabled,
                outlier_filter_report=read_result.filter_report,
            ),
        )
        if should_save:
            if target_output_dir is None:
                raise ValueError("Saving requested but no output_dir was provided")
            self.save(output, target_output_dir)
        return output

    def _read_input(
        self,
        input_root: str | os.PathLike[str] | Mapping[str, np.ndarray],
        scene_name: str | None,
        outlier_filter: bool | Mapping[str, Any] | None,
    ) -> GaussianReadResult:
        if isinstance(input_root, Mapping):
            data = {key: np.asarray(value) if isinstance(value, np.ndarray) else value for key, value in input_root.items()}
            resolved_name = scene_name or str(data.get("name", self.default_scene_name))
            data["name"] = resolved_name
            row_count = int(np.asarray(data["coord"]).shape[0])
            return GaussianReadResult(
                data=data,
                input_path="<array>",
                scene_name=resolved_name,
                source_type="array",
                raw_count=row_count,
                kept_count=row_count,
                outlier_filter_enabled=False,
            )

        path = Path(input_root).expanduser()
        if path.suffix.lower() == ".ply" and not self.cfg.get("raw_ply", False):
            raise ValueError(
                "Raw .ply input is unsupported for chorus_pts. Use a scene folder "
                "with coord.npy, color.npy, and normal.npy."
            )

        reader_cfg = dict(self.cfg.get("input_reader", {}))
        if outlier_filter is not None:
            reader_cfg = json.loads(json.dumps(reader_cfg))
            current = reader_cfg.setdefault("outlier_filter", {})
            if isinstance(outlier_filter, bool):
                current["enabled"] = outlier_filter
            else:
                current.update(dict(outlier_filter))
        return load_gaussian_input(
            path,
            self.feat_keys,
            scene_name=scene_name,
            input_reader_cfg=reader_cfg,
            logger=self.logger,
        )

    def _encode_arrays(
        self,
        data: Mapping[str, np.ndarray],
        *,
        scene_name: str,
        metadata: dict[str, Any],
    ) -> ChorusOutput:
        prepared = self._prepare_input_dict(data, scene_name)
        fragments = prepared.pop("fragment_list")
        scene = prepared.get("name", scene_name or self.default_scene_name)
        num_points = (
            prepared["segment"].shape[0]
            if prepared.get("segment") is not None
            else fragments[0]["coord"].shape[0]
        )
        accumulators: dict[str, torch.Tensor] = {}
        counts: dict[str, torch.Tensor] = {}
        raw_token_parts = []

        for fragment_id, fragment in enumerate(fragments):
            input_dict = collate_fn([fragment])
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(self.device, non_blocking=True)

            with torch.no_grad():
                out_dict = self.model(
                    input_dict,
                    chunk_size=self.chunk_size,
                    teacher_names=self.active_teachers,
                    return_backbone_upcast="backbone_upcast" in self.outputs,
                    return_backbone_last="backbone_last" in self.outputs,
                )

            aligned_outputs = dict(out_dict["point_feat"])
            if "backbone_upcast" in out_dict:
                aligned_outputs["backbone_upcast"] = out_dict["backbone_upcast"]
            self._accumulate_aligned(
                aligned_outputs,
                input_dict["index"],
                input_dict["offset"],
                num_points,
                accumulators,
                counts,
            )

            if "backbone_last" in out_dict:
                raw_token_parts.append(
                    self._token_part_to_cpu(out_dict["backbone_last"], fragment_id)
                )

        inverse_map = prepared.get("inverse")
        features = self._finalize_aligned(accumulators, counts, inverse_map)
        tokens = {}
        if raw_token_parts:
            tokens["backbone_last"] = self._finalize_tokens(raw_token_parts)

        output_metadata = {
            "origin_coord": prepared.get("origin_coord"),
            "origin_segment": prepared.get("origin_segment"),
            "origin_feat_mask": prepared.get("origin_feat_mask"),
            "inverse": inverse_map,
            **metadata,
        }
        return ChorusOutput(name=scene, features=features, tokens=tokens, metadata=output_metadata)

    @staticmethod
    def _accumulate_aligned(
        feature_map: Mapping[str, torch.Tensor],
        idx_part: torch.Tensor,
        offset_list: torch.Tensor,
        num_points: int,
        accumulators: dict[str, torch.Tensor],
        counts: dict[str, torch.Tensor],
    ) -> None:
        start = 0
        for end in offset_list:
            slice_idx = idx_part[start:end]
            for name, feat in feature_map.items():
                acc = accumulators.get(name)
                cnt = counts.get(name)
                if acc is None:
                    acc = torch.zeros(
                        (num_points, feat.size(1)),
                        device=feat.device,
                        dtype=feat.dtype,
                    )
                    cnt = torch.zeros(num_points, device=feat.device, dtype=feat.dtype)
                    accumulators[name] = acc
                    counts[name] = cnt
                acc[slice_idx] += feat[start:end]
                cnt[slice_idx] += 1
            start = end

    def _finalize_aligned(
        self,
        accumulators: Mapping[str, torch.Tensor],
        counts: Mapping[str, torch.Tensor],
        inverse_map: np.ndarray | None,
    ) -> dict[str, Any]:
        features = {}
        for name, acc in accumulators.items():
            cnt = counts[name]
            valid_mask = cnt > 0
            if torch.any(valid_mask):
                acc[valid_mask] = acc[valid_mask] / cnt[valid_mask].unsqueeze(1)
            if inverse_map is not None:
                inverse_tensor = torch.as_tensor(inverse_map, device=acc.device, dtype=torch.long)
                acc = acc[inverse_tensor]
            if name in TEACHER_OUTPUTS:
                acc = F.normalize(acc, p=2, dim=1)
            acc_cpu = acc.detach().cpu()
            features[name] = acc_cpu.numpy() if self.return_numpy else acc_cpu
        return features

    @staticmethod
    def _token_part_to_cpu(tokens: Mapping[str, torch.Tensor], fragment_id: int) -> dict[str, torch.Tensor]:
        result = {key: value.detach().cpu() for key, value in tokens.items()}
        count = int(result["feat"].shape[0])
        result["fragment_index"] = torch.full((count,), fragment_id, dtype=torch.long)
        return result

    def _finalize_tokens(self, parts: list[dict[str, torch.Tensor]]) -> TokenOutput:
        values = {}
        for key in ("feat", "coord", "grid_coord", "fragment_index"):
            tensors = [part[key] for part in parts if key in part]
            values[key] = torch.cat(tensors, dim=0) if tensors else None
        offsets = []
        total = 0
        for part in parts:
            total += int(part["feat"].shape[0])
            offsets.append(total)
        values["offset"] = torch.tensor(offsets, dtype=torch.long)
        if self.return_numpy:
            for key, value in list(values.items()):
                if isinstance(value, torch.Tensor):
                    values[key] = value.numpy()
        return TokenOutput(**values)

    def _prepare_input_dict(
        self, data: Mapping[str, np.ndarray], scene_name: str | None
    ) -> dict[str, Any]:
        base_dict = self._format_numpy_inputs(data, scene_name)
        data_dict = self.transform(base_dict)
        result = dict(
            segment=data_dict.pop("segment", None),
            name=data_dict.pop("name", scene_name or self.default_scene_name),
        )
        for key in ("coord", "pc_coord", "pc_segment"):
            if key in data_dict:
                result[key] = data_dict[key]
        for key in (
            "origin_coord",
            "origin_feat_mask",
            "origin_instance",
            "origin_dino_feat",
            "origin_segment",
            "inverse",
        ):
            if key in data_dict:
                result[key] = data_dict.pop(key)

        fragments = [aug(copy.deepcopy(data_dict)) for aug in self.aug_transform]
        fragment_list = []
        for fragment in fragments:
            data_slices = self.test_voxelize(fragment) if self.test_voxelize else [fragment]
            if self.test_voxelize is None:
                fragment["index"] = np.arange(fragment["coord"].shape[0])
            for slice_data in data_slices:
                crop_list = self.test_crop(slice_data) if self.test_crop else [slice_data]
                fragment_list.extend(crop_list)
        result["fragment_list"] = [self.post_transform(frag) for frag in fragment_list]
        return result

    def _format_numpy_inputs(
        self, data: Mapping[str, np.ndarray], scene_name: str | None
    ) -> dict[str, Any]:
        missing = [key for key in self.feat_keys if key not in data]
        if missing:
            raise KeyError("Missing required features: " + ", ".join(sorted(set(missing))))

        formatted: dict[str, Any] = {}
        num_points = None
        if "coord" in data:
            coord = np.asarray(data["coord"], dtype=np.float32)
            formatted["coord"] = coord
            num_points = coord.shape[0]
        if num_points is None:
            reference = np.asarray(data[self.feat_keys[0]])
            num_points = reference.shape[0]

        def ensure_num_points(name: str, array: np.ndarray) -> np.ndarray:
            if array.shape[0] != num_points:
                raise ValueError(f"{name} shape mismatch: expected {num_points} rows.")
            return array

        for key in ("color", "quat", "scale", "normal"):
            if key in data:
                formatted[key] = ensure_num_points(key, np.asarray(data[key], dtype=np.float32))
        if "opacity" in data:
            opacity = np.asarray(data["opacity"], dtype=np.float32)
            if opacity.ndim == 1:
                opacity = opacity.reshape(-1, 1)
            formatted["opacity"] = ensure_num_points("opacity", opacity)

        if "segment" in data:
            segment = np.asarray(data["segment"])
            if segment.ndim == 2:
                segment = segment[:, 0]
            formatted["segment"] = ensure_num_points(
                "segment", segment.reshape(-1).astype(np.int32)
            )
        else:
            formatted["segment"] = np.full((num_points,), -1, dtype=np.int32)
        if "instance" in data:
            instance = np.asarray(data["instance"])
            if instance.ndim == 2:
                instance = instance[:, 0]
            formatted["instance"] = ensure_num_points(
                "instance", instance.reshape(-1).astype(np.int32)
            )
        else:
            formatted["instance"] = np.full((num_points,), -1, dtype=np.int32)
        formatted["name"] = scene_name or data.get("name", self.default_scene_name)
        return formatted

    def save(self, output: ChorusOutput, output_dir: str | os.PathLike[str]) -> None:
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        source_keep_index = output.metadata.get("source_keep_index")
        for name, feature in output.features.items():
            file_path = output_path / f"{output.name}_{name}_feat.pt"
            tensor = self._as_tensor(feature).half()
            torch.save(tensor, file_path)
            self.logger.info("Saved %s features to %s", name, file_path)
            if source_keep_index is not None:
                index_path = file_path.with_name(f"{file_path.stem}_index.npy")
                np.save(index_path, np.asarray(source_keep_index, dtype=np.int64))
                self.logger.info("Saved kept-splat index sidecar to %s", index_path)
        for name, token_output in output.tokens.items():
            file_path = output_path / f"{output.name}_{name}_tokens.pt"
            torch.save(
                {
                    key: self._as_tensor(value) if value is not None else None
                    for key, value in token_output.__dict__.items()
                },
                file_path,
            )
            self.logger.info("Saved %s tokens to %s", name, file_path)

    @staticmethod
    def _as_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.as_tensor(value)
