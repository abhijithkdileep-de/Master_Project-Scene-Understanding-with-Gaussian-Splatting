"""PCA colorize SceneSplat feature tensors for preprocessed 3DGS scenes."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SH_C0 = 0.28209479177387814
_DEFAULT_PCA_SEED = 1219
_PCA_Q = 6
_PCA_PERCENTILE_LOW = 1.0
_PCA_PERCENTILE_HIGH = 99.0
PCA_METHOD_PC123 = "pc123_pct01_99_q6"
PCA_METHOD_BASELINE_MIX = "baseline_mix075_minmax_q6"
PCA_METHODS = (PCA_METHOD_PC123, PCA_METHOD_BASELINE_MIX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        required=True,
        help="Saved feature tensor path (.pt/.pth/.npy/.npz).",
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Preprocessed scene folder containing coord.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PCA visualization outputs. Defaults to <repo_root>/outputs.",
    )
    parser.add_argument("--scene-name", default=None, help="Optional scene name override.")
    parser.add_argument(
        "--pca-method",
        choices=PCA_METHODS,
        default=PCA_METHOD_PC123,
        help=f"PCA color mapping. Defaults to {PCA_METHOD_PC123}.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for torch PCA coloring. Defaults to cpu.",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=1.25,
        help="Brightness multiplier applied before clamping colors to [0, 1].",
    )
    parser.add_argument(
        "--pca-seed",
        type=int,
        default=_DEFAULT_PCA_SEED,
        help=f"Seed for PCA colorization. Defaults to {_DEFAULT_PCA_SEED}.",
    )
    parser.add_argument(
        "--pca-niter",
        type=int,
        default=5,
        help="Number of torch.pca_lowrank subspace iterations.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("scenesplat.pca")
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


def infer_scene_name(input_root: str | Path, explicit_name: Optional[str] = None) -> str:
    if explicit_name:
        return explicit_name
    return Path(input_root).expanduser().resolve().name


def get_default_output_dir() -> Path:
    return _REPO_ROOT / "outputs"


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


def _to_numpy_feature_array(obj, source_path: str | Path) -> np.ndarray:
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
            value = obj.get(key)
            if isinstance(value, (torch.Tensor, np.ndarray)):
                obj = value
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


def _load_required_npy(scene_root: Path, name: str) -> np.ndarray:
    path = scene_root / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing required scene array: {path}")
    return np.asarray(np.load(path, mmap_mode="r"))


def _load_optional_npy(scene_root: Path, name: str) -> Optional[np.ndarray]:
    path = scene_root / f"{name}.npy"
    if not path.exists():
        return None
    return np.asarray(np.load(path, mmap_mode="r"))


def _require_rows(name: str, array: np.ndarray, rows: int) -> np.ndarray:
    if array.shape[0] != rows:
        raise ValueError(f"{name}.npy row count {array.shape[0]} does not match coord rows {rows}")
    return array


def compute_pca_colors(
    features: np.ndarray,
    *,
    method: str,
    device: str,
    brightness: float,
    pca_seed: int,
    pca_niter: int,
) -> np.ndarray:
    import torch

    if method not in PCA_METHODS:
        raise ValueError(f"Unsupported PCA method: {method}")
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Feature array contains NaN or Inf values")

    set_seed(pca_seed)
    feat = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    q_eff = min(_PCA_Q, feat.shape[0], feat.shape[1])
    min_components = 6 if method == PCA_METHOD_BASELINE_MIX else 3
    if q_eff < min_components:
        raise ValueError(
            f"{method} requires at least {min_components} PCA components, "
            f"got q={q_eff} for shape={tuple(feat.shape)}"
        )

    with torch.no_grad():
        _, _, v = torch.pca_lowrank(feat, center=True, q=q_eff, niter=pca_niter)
        projection = feat @ v
        if method == PCA_METHOD_PC123:
            mixed = projection[:, :3]
            low = torch.quantile(
                mixed, _PCA_PERCENTILE_LOW / 100.0, dim=0, keepdim=True
            )
            high = torch.quantile(
                mixed, _PCA_PERCENTILE_HIGH / 100.0, dim=0, keepdim=True
            )
            colors = (mixed - low) / torch.clamp(high - low, min=1e-6)
        else:
            mixed = projection[:, :3] * 0.75 + projection[:, 3:6] * 0.25
            low = mixed.amin(dim=0, keepdim=True)
            high = mixed.amax(dim=0, keepdim=True)
            colors = (mixed - low) / torch.clamp(high - low, min=1e-6)
        colors = (colors * brightness).clamp_(0.0, 1.0).cpu().numpy()

    return colors.astype(np.float32, copy=False)


def write_point_cloud(coord: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), pcd):
        raise RuntimeError(f"Open3D failed to write {output_path}")


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
    if colors_01.shape != (num_points, 3):
        raise ValueError(
            f"colors_01 shape {colors_01.shape} does not match expected {(num_points, 3)}"
        )
    if normal is None:
        normal = np.zeros_like(coord, dtype=np.float32)

    opacity = opacity.reshape(-1, 1).astype(np.float32, copy=False)
    scale = scale.reshape(scale.shape[0], -1).astype(np.float32, copy=False)
    quat = quat.reshape(quat.shape[0], -1).astype(np.float32, copy=False)
    normal = normal.reshape(normal.shape[0], -1).astype(np.float32, copy=False)
    for name, array in (
        ("opacity", opacity),
        ("scale", scale),
        ("quat", quat),
        ("normal", normal),
    ):
        if array.shape[0] != num_points:
            raise ValueError(f"{name} rows {array.shape[0]} do not match coord rows {num_points}")
    if normal.shape[1] < 3:
        raise ValueError(f"normal must have at least 3 columns, got shape={normal.shape}")

    f_dc = ((colors_01.astype(np.float32, copy=False) - 0.5) / _SH_C0).astype(np.float32)
    raw_opacity = _logit(opacity)
    raw_scale = np.log(np.maximum(scale, 1e-7)).astype(np.float32, copy=False)
    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(quat_norm, 1e-7)

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
    pca_method: str = PCA_METHOD_PC123,
    device: str = "cpu",
    brightness: float = 1.25,
    pca_seed: int = _DEFAULT_PCA_SEED,
    pca_niter: int = 5,
    logger: Optional[logging.Logger] = None,
) -> dict:
    logger = logger or setup_logger()
    feature_path = Path(feature_path).expanduser()
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature path does not exist: {feature_path}")

    feature_obj = _load_serialized_array(feature_path)
    features = _to_numpy_feature_array(feature_obj, feature_path)
    return _run_pca_visualization_from_features(
        features=features,
        input_root=input_root,
        output_dir=output_dir,
        scene_name=scene_name,
        pca_method=pca_method,
        device=device,
        brightness=brightness,
        pca_seed=pca_seed,
        pca_niter=pca_niter,
        logger=logger,
        feature_source="saved",
        feature_path=feature_path,
    )


def run_pca_visualization_from_features(
    *,
    features,
    input_root: str | Path,
    output_dir: str | Path | None = None,
    scene_name: Optional[str] = None,
    pca_method: str = PCA_METHOD_PC123,
    device: str = "cpu",
    brightness: float = 1.25,
    pca_seed: int = _DEFAULT_PCA_SEED,
    pca_niter: int = 5,
    logger: Optional[logging.Logger] = None,
) -> dict:
    return _run_pca_visualization_from_features(
        features=features,
        input_root=input_root,
        output_dir=output_dir,
        scene_name=scene_name,
        pca_method=pca_method,
        device=device,
        brightness=brightness,
        pca_seed=pca_seed,
        pca_niter=pca_niter,
        logger=logger or setup_logger(),
        feature_source="memory",
        feature_path=None,
    )


def _run_pca_visualization_from_features(
    *,
    features,
    input_root: str | Path,
    output_dir: str | Path | None,
    scene_name: Optional[str],
    pca_method: str,
    device: str,
    brightness: float,
    pca_seed: int,
    pca_niter: int,
    logger: logging.Logger,
    feature_source: str,
    feature_path: Optional[Path],
) -> dict:
    input_root = Path(input_root).expanduser()
    output_dir = (
        Path(output_dir).expanduser() if output_dir is not None else get_default_output_dir()
    )
    scene_name = infer_scene_name(input_root, explicit_name=scene_name)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root must be a preprocessed scene folder: {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    coord = _load_required_npy(input_root, "coord").astype(np.float32, copy=False)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"coord.npy must have shape (N, 3), got {coord.shape}")

    features = _to_numpy_feature_array(features, f"{feature_source} features")
    if features.shape[0] != coord.shape[0]:
        raise ValueError(
            f"Feature rows {features.shape[0]} do not match coord rows {coord.shape[0]}"
        )

    logger.info(
        "Computing PCA colors: scene=%s features=%s source=%s method=%s device=%s seed=%d",
        scene_name,
        tuple(features.shape),
        feature_source,
        pca_method,
        device,
        pca_seed,
    )
    colors = compute_pca_colors(
        features,
        method=pca_method,
        device=device,
        brightness=brightness,
        pca_seed=pca_seed,
        pca_niter=pca_niter,
    )

    point_cloud_path = output_dir / f"{scene_name}_pca_colored.ply"
    write_point_cloud(coord, colors, point_cloud_path)
    logger.info("Saved PCA point cloud to %s", point_cloud_path)

    opacity = _load_optional_npy(input_root, "opacity")
    scale = _load_optional_npy(input_root, "scale")
    quat = _load_optional_npy(input_root, "quat")
    normal = _load_optional_npy(input_root, "normal")
    featvis_path = None
    if opacity is not None and scale is not None and quat is not None:
        rows = coord.shape[0]
        opacity = _require_rows("opacity", opacity, rows)
        scale = _require_rows("scale", scale, rows)
        quat = _require_rows("quat", quat, rows)
        if normal is not None:
            normal = _require_rows("normal", normal, rows)
        featvis_path = output_dir / f"{scene_name}_feat_vis_3dgs.ply"
        write_featvis_3dgs_ply(
            featvis_path,
            coord=coord,
            colors_01=colors,
            opacity=opacity,
            scale=scale,
            quat=quat,
            normal=normal,
        )
        logger.info("Saved feat-vis 3DGS PLY to %s", featvis_path)
    else:
        logger.warning(
            "Skipping feat-vis 3DGS export because opacity.npy, scale.npy, or quat.npy is missing."
        )

    return dict(
        scene_name=scene_name,
        output_dir=str(output_dir),
        feature_source=feature_source,
        feature_path=str(feature_path) if feature_path is not None else None,
        pca_method=pca_method,
        point_cloud_path=str(point_cloud_path),
        featvis_path=str(featvis_path) if featvis_path is not None else None,
    )


def main() -> None:
    args = parse_args()
    run_pca_visualization(
        feature_path=args.feature_path,
        input_root=args.input_root,
        output_dir=args.output_dir,
        scene_name=args.scene_name,
        pca_method=args.pca_method,
        device=args.device,
        brightness=args.brightness,
        pca_seed=args.pca_seed,
        pca_niter=args.pca_niter,
    )


if __name__ == "__main__":
    main()
