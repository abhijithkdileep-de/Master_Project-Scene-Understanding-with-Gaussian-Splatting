"""PCA colorize Chorus feature outputs and optionally write a feat-vis 3DGS PLY."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scripts.gaussian_io import (
        get_default_output_dir,
        infer_scene_name,
        load_gaussian_input,
    )
except ImportError:
    from gaussian_io import get_default_output_dir, infer_scene_name, load_gaussian_input

_SH_C0 = 0.28209479177387814
_DEFAULT_PCA_SEED = 1
_PCA_COLOR_METHOD = "pc123_pct01_99"
_PCA_COLOR_METHODS = ("pc123_pct01_99", "mix_minmax_q6")
_PCA_PC123_Q = 3
_PCA_MIX_Q = 6
_PCA_MIX_PRIMARY_WEIGHT = 0.7
_PCA_PERCENTILE_LOW = 1.0
_PCA_PERCENTILE_HIGH = 99.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", required=True, help="Saved feature tensor (.pt/.pth/.npy/.npz).")
    parser.add_argument(
        "--input-root",
        required=True,
        help="Original scene directory or raw/compressed .ply used for inference.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PCA visualization outputs. Defaults to <repo_root>/outputs.",
    )
    parser.add_argument("--scene-name", default=None, help="Optional scene name override.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for torch PCA coloring. Defaults to cpu.",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=1.25,
        help="Brightness multiplier after PCA color normalization.",
    )
    parser.add_argument(
        "--pca-color-method",
        choices=_PCA_COLOR_METHODS,
        default=_PCA_COLOR_METHOD,
        help=f"PCA colorization method. Defaults to {_PCA_COLOR_METHOD}.",
    )
    parser.add_argument(
        "--incremental-pca-batch-size",
        type=int,
        default=500_000,
        help="Batch size for IncrementalPCA fallback.",
    )
    parser.add_argument(
        "--pca-seed",
        type=int,
        default=_DEFAULT_PCA_SEED,
        help=f"Seed for PCA colorization randomness. Defaults to {_DEFAULT_PCA_SEED}.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("chorus.pca")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_serialized_array(path: Path):
    import torch

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".ckpt"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".npz":
        npz_obj = np.load(path, allow_pickle=True)
        if "features" in npz_obj:
            return npz_obj["features"]
        if "arr_0" in npz_obj:
            return npz_obj["arr_0"]
        first_key = next(iter(npz_obj.files), None)
        if first_key is None:
            raise ValueError(f"Empty .npz feature file: {path}")
        return npz_obj[first_key]
    return np.load(path, allow_pickle=True)


def _to_numpy_feature_array(obj, source_path: Path) -> np.ndarray:
    import torch

    if isinstance(obj, (tuple, list)):
        for item in obj:
            if isinstance(item, (torch.Tensor, np.ndarray)):
                obj = item
                break
        else:
            raise TypeError(f"No tensor/ndarray found in sequence loaded from {source_path}")

    if isinstance(obj, dict):
        for key in ("features", "feats", "feat", "embedding", "embeddings"):
            if key in obj and isinstance(obj[key], (torch.Tensor, np.ndarray)):
                obj = obj[key]
                break
        else:
            for value in obj.values():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    obj = value
                    break
            else:
                raise TypeError(f"No tensor/ndarray found in dict loaded from {source_path}")

    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        arr = np.asarray(obj)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={arr.shape} from {source_path}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Loaded features are not numeric from {source_path}")
    return arr.astype(np.float32, copy=False)


def _resolve_index_sidecar(feature_path: Path) -> Optional[Path]:
    candidate = feature_path.with_name(f"{feature_path.stem}_index.npy")
    if candidate.exists():
        return candidate
    return None


def _load_index_array(index_path: Path) -> np.ndarray:
    index = np.load(index_path)
    if index.ndim != 1:
        raise ValueError(f"Expected 1D index, got shape={index.shape} from {index_path}")
    if not np.issubdtype(index.dtype, np.integer):
        raise TypeError(f"Expected integer indices in {index_path}, got {index.dtype}")
    return index.astype(np.int64, copy=False)


def _validate_pca_color_method(pca_color_method: str) -> str:
    if pca_color_method not in _PCA_COLOR_METHODS:
        raise ValueError(
            f"Unsupported PCA color method {pca_color_method!r}; "
            f"expected one of {', '.join(_PCA_COLOR_METHODS)}"
        )
    return pca_color_method


def _pca_rank_for_method(pca_color_method: str) -> int:
    pca_color_method = _validate_pca_color_method(pca_color_method)
    if pca_color_method == "mix_minmax_q6":
        return _PCA_MIX_Q
    return _PCA_PC123_Q


def get_pca_color_torch(
    feat,
    *,
    brightness: float,
    pca_color_method: str = _PCA_COLOR_METHOD,
    center: bool = True,
    q: Optional[int] = None,
    niter: int = 5,
):
    import torch

    pca_color_method = _validate_pca_color_method(pca_color_method)
    q_target = _pca_rank_for_method(pca_color_method) if q is None else q
    q_eff = min(q_target, feat.shape[0], feat.shape[1])
    if q_eff < 3:
        raise ValueError(f"Need at least 3 PCA components, got q={q_eff} for shape={tuple(feat.shape)}")
    _, _, v = torch.pca_lowrank(feat, center=center, q=q_eff, niter=niter)
    projection = feat @ v
    if pca_color_method == "mix_minmax_q6":
        if projection.shape[1] >= _PCA_MIX_Q:
            mix = (
                projection[:, :3] * _PCA_MIX_PRIMARY_WEIGHT
                + projection[:, 3:6] * (1.0 - _PCA_MIX_PRIMARY_WEIGHT)
            )
        else:
            mix = projection[:, :3]
        low = mix.amin(dim=0, keepdim=True)
        high = mix.amax(dim=0, keepdim=True)
    else:
        mix = projection[:, :3]
        low = torch.quantile(mix, _PCA_PERCENTILE_LOW / 100.0, dim=0, keepdim=True)
        high = torch.quantile(mix, _PCA_PERCENTILE_HIGH / 100.0, dim=0, keepdim=True)
    color = (mix - low) / torch.clamp(high - low, min=1e-6)
    return (color * brightness).clamp_(0.0, 1.0)


def get_pca_color_sklearn(
    feat: np.ndarray,
    *,
    brightness: float,
    batch_size: int,
    pca_color_method: str = _PCA_COLOR_METHOD,
) -> np.ndarray:
    from sklearn.decomposition import IncrementalPCA, PCA

    pca_color_method = _validate_pca_color_method(pca_color_method)
    q_eff = min(_pca_rank_for_method(pca_color_method), feat.shape[0], feat.shape[1])
    if q_eff < 3:
        raise ValueError(f"Need at least 3 PCA components, got q={q_eff} for shape={feat.shape}")
    if feat.shape[0] > 100_000:
        pca = IncrementalPCA(n_components=q_eff, batch_size=max(batch_size, q_eff))
        pca.fit(feat)
    else:
        pca = PCA(n_components=q_eff)
        pca.fit(feat)
    feat_pca = (feat @ pca.components_.T).astype(np.float32, copy=False)
    if pca_color_method == "mix_minmax_q6":
        if feat_pca.shape[1] >= _PCA_MIX_Q:
            mix = (
                feat_pca[:, :3] * _PCA_MIX_PRIMARY_WEIGHT
                + feat_pca[:, 3:6] * (1.0 - _PCA_MIX_PRIMARY_WEIGHT)
            )
        else:
            mix = feat_pca[:, :3]
        low = mix.min(axis=0, keepdims=True)
        high = mix.max(axis=0, keepdims=True)
    else:
        mix = feat_pca[:, :3]
        low = np.percentile(mix, _PCA_PERCENTILE_LOW, axis=0, keepdims=True)
        high = np.percentile(mix, _PCA_PERCENTILE_HIGH, axis=0, keepdims=True)
    color = (mix - low) / np.maximum(high - low, 1e-6)
    return np.clip(color * brightness, 0.0, 1.0).astype(np.float32)


def write_point_cloud(coords: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)


def _logit(value: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    value = np.clip(value, eps, 1.0 - eps)
    return np.log(value / (1.0 - value))


def write_featvis_3dgs_ply(
    output_path: Path,
    *,
    coord: np.ndarray,
    colors_01: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    quat: np.ndarray,
    normal: Optional[np.ndarray] = None,
    max_sh_degree: int = 3,
) -> None:
    from plyfile import PlyData, PlyElement

    num_points = coord.shape[0]
    if normal is None:
        normal = np.zeros_like(coord, dtype=np.float32)

    f_dc = ((colors_01.astype(np.float32) - 0.5) / _SH_C0).astype(np.float32)
    raw_opacity = _logit(opacity.reshape(-1, 1).astype(np.float32))
    raw_scale = np.log(np.maximum(scale.astype(np.float32), 1e-7))
    quat = quat.astype(np.float32)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-7)

    num_f_rest = 3 * ((max_sh_degree + 1) ** 2 - 1)
    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    dtype_list.extend((f"f_rest_{idx}", "f4") for idx in range(num_f_rest))
    dtype_list.append(("opacity", "f4"))
    dtype_list.extend((f"scale_{idx}", "f4") for idx in range(raw_scale.shape[1]))
    dtype_list.extend((f"rot_{idx}", "f4") for idx in range(quat.shape[1]))

    vertex = np.empty(num_points, dtype=dtype_list)
    vertex["x"] = coord[:, 0]
    vertex["y"] = coord[:, 1]
    vertex["z"] = coord[:, 2]
    vertex["nx"] = normal[:, 0]
    vertex["ny"] = normal[:, 1]
    vertex["nz"] = normal[:, 2]
    vertex["f_dc_0"] = f_dc[:, 0]
    vertex["f_dc_1"] = f_dc[:, 1]
    vertex["f_dc_2"] = f_dc[:, 2]
    for idx in range(num_f_rest):
        vertex[f"f_rest_{idx}"] = 0.0
    vertex["opacity"] = raw_opacity[:, 0]
    for idx in range(raw_scale.shape[1]):
        vertex[f"scale_{idx}"] = raw_scale[:, idx]
    for idx in range(quat.shape[1]):
        vertex[f"rot_{idx}"] = quat[:, idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output_path))


def run_pca_visualization(
    *,
    feature_path: str | Path,
    input_root: str | Path,
    output_dir: str | Path | None = None,
    scene_name: Optional[str] = None,
    device: str = "cpu",
    brightness: float = 1.25,
    pca_color_method: str = _PCA_COLOR_METHOD,
    incremental_pca_batch_size: int = 500_000,
    pca_seed: int = _DEFAULT_PCA_SEED,
    logger: Optional[logging.Logger] = None,
) -> dict:
    import torch

    logger = logger or setup_logger()
    feature_path = Path(feature_path)
    output_dir = Path(output_dir) if output_dir is not None else get_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_name = infer_scene_name(input_root, explicit_name=scene_name)
    pca_color_method = _validate_pca_color_method(pca_color_method)

    feature_obj = _load_serialized_array(feature_path)
    feat = _to_numpy_feature_array(feature_obj, feature_path)
    if not np.isfinite(feat).all():
        raise ValueError(f"Feature array contains NaN/Inf values: {feature_path}")
    logger.info("Loaded features %s with shape=%s", feature_path, tuple(feat.shape))

    index_path = _resolve_index_sidecar(feature_path)
    kept_index = _load_index_array(index_path) if index_path is not None else None
    if kept_index is not None:
        logger.info("Loaded feature index %s with shape=%s", index_path, tuple(kept_index.shape))

    source = load_gaussian_input(
        input_root,
        required_keys=(),
        optional_keys=("opacity", "scale", "quat", "normal"),
        scene_name=scene_name,
        input_reader_cfg=dict(outlier_filter=dict(enabled=False)),
        logger=logger,
    )
    coord_full = source.data["coord"].astype(np.float32, copy=False)

    if kept_index is not None:
        if feat.shape[0] != kept_index.shape[0]:
            raise ValueError(
                f"Feature rows ({feat.shape[0]}) do not match kept-index rows ({kept_index.shape[0]})"
            )
        if np.any(kept_index < 0) or np.any(kept_index >= coord_full.shape[0]):
            raise ValueError("Feature indices fall outside the source splat range")
        coord = coord_full[kept_index]
    else:
        if feat.shape[0] != coord_full.shape[0]:
            raise ValueError(
                "Feature rows do not match source splat count and no *_index.npy was found."
            )
        coord = coord_full

    set_seed(pca_seed)
    try:
        feat_t = torch.from_numpy(feat).to(torch.float32).to(device)
        with torch.no_grad():
            colors = get_pca_color_torch(
                feat_t,
                brightness=brightness,
                pca_color_method=pca_color_method,
            ).cpu().numpy()
        logger.info(
            "Computed PCA colors with torch.pca_lowrank on %s (seed=%d, method=%s)",
            device,
            pca_seed,
            pca_color_method,
        )
    except Exception as exc:
        logger.warning(
            "Torch PCA failed (%s). Falling back to sklearn PCA (method=%s).",
            exc,
            pca_color_method,
        )
        set_seed(pca_seed)
        colors = get_pca_color_sklearn(
            feat,
            brightness=brightness,
            batch_size=incremental_pca_batch_size,
            pca_color_method=pca_color_method,
        )

    point_cloud_path = output_dir / f"{scene_name}_pca_colored.ply"
    write_point_cloud(coord, colors, point_cloud_path)
    logger.info("Saved PCA point cloud to %s", point_cloud_path)

    required_for_3dgs = {"opacity", "scale", "quat"}
    featvis_path = None
    if required_for_3dgs.issubset(source.data.keys()):
        selector = kept_index if kept_index is not None else slice(None)
        normal = source.data.get("normal")
        featvis_path = output_dir / f"{scene_name}_feat_vis_3dgs.ply"
        write_featvis_3dgs_ply(
            featvis_path,
            coord=coord,
            colors_01=colors,
            opacity=np.asarray(source.data["opacity"])[selector],
            scale=np.asarray(source.data["scale"])[selector],
            quat=np.asarray(source.data["quat"])[selector],
            normal=np.asarray(normal)[selector] if normal is not None else None,
        )
        logger.info("Saved feat-vis 3DGS PLY to %s", featvis_path)
    else:
        logger.warning(
            "Skipping feat-vis 3DGS export because the source input does not contain opacity/scale/quat."
        )

    return dict(
        scene_name=scene_name,
        output_dir=str(output_dir),
        feature_path=str(feature_path),
        pca_color_method=pca_color_method,
        point_cloud_path=str(point_cloud_path),
        featvis_path=str(featvis_path) if featvis_path is not None else None,
        index_path=str(index_path) if index_path is not None else None,
    )


def main() -> None:
    args = parse_args()
    run_pca_visualization(
        feature_path=args.feature_path,
        input_root=args.input_root,
        output_dir=args.output_dir,
        scene_name=args.scene_name,
        device=args.device,
        brightness=args.brightness,
        pca_color_method=args.pca_color_method,
        incremental_pca_batch_size=args.incremental_pca_batch_size,
        pca_seed=args.pca_seed,
    )


if __name__ == "__main__":
    main()
