from pathlib import Path
import shutil

import numpy as np
import torch


TEACHER_FEATURE_KEYS = {
    "lang_feat.npy",
    "lang_feat_index.npy",
    "valid_feat_mask.npy",
    "dino_feat.npy",
    "dino_feat_index.npy",
    "pe_feat.npy",
    "pe_feat_index.npy",
}


def find_npy_scene_dir(root, scene_name, split_name=None):
    if root is None:
        return None

    root = Path(root)
    candidates = []
    if split_name is not None:
        candidates.append(root / split_name / scene_name)
    candidates.extend(
        [
            root / scene_name,
            root / "train" / scene_name,
            root / "val" / scene_name,
            root / "test" / scene_name,
            root / "test_eval" / scene_name,
            root / "train_v1" / scene_name,
        ]
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "coord.npy").exists():
            return candidate
    return None


def copy_base_npy_scene(scene_dir, save_path, exclude_teacher_features=True):
    scene_dir = Path(scene_dir)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    copied = []
    for path in scene_dir.iterdir():
        if path.suffix != ".npy":
            continue
        if exclude_teacher_features and path.name in TEACHER_FEATURE_KEYS:
            continue
        shutil.copy2(path, save_path / path.name)
        copied.append(path.name)
    return copied


def write_compact_teacher_features(
    scene_name,
    save_path,
    coord_count,
    feat_root=None,
    dino_root=None,
    pe_root=None,
):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    within_mask = np.arange(coord_count)

    written = {}
    if feat_root is not None:
        feat_path, _, _ = resolve_occamlgs_feature_paths(feat_root, scene_name)
        if feat_path is not None and feat_path.exists():
            lang_feat, lang_index = load_occamlgs_feature(feat_root, scene_name)
            local_index, _ = save_compact_feature(
                lang_feat,
                lang_index,
                within_mask,
                save_path,
                "lang_feat",
                write_valid_mask=True,
            )
            written["lang_feat"] = len(local_index)

    if dino_root is not None:
        dino_path, dino_index_path = resolve_ludvig_feature_paths(
            dino_root, scene_name, "dinov3", "dino_feat"
        )
        if dino_path is not None and dino_path.exists():
            dino_feat, dino_index = load_npy_feature(dino_path, dino_index_path)
            local_index, _ = save_compact_feature(
                dino_feat,
                dino_index,
                within_mask,
                save_path,
                "dino_feat",
            )
            written["dino_feat"] = len(local_index)

    if pe_root is not None:
        pe_path, pe_index_path = resolve_ludvig_feature_paths(
            pe_root, scene_name, "pe_spatial", "pe_feat"
        )
        if pe_path is not None and pe_path.exists():
            pe_feat, pe_index = load_npy_feature(pe_path, pe_index_path)
            local_index, _ = save_compact_feature(
                pe_feat,
                pe_index,
                within_mask,
                save_path,
                "pe_feat",
            )
            written["pe_feat"] = len(local_index)

    return written


def load_torch_feature(feature_path, index_path=None):
    feature = _torch_load_first(feature_path)
    if isinstance(feature, torch.Tensor):
        feature = feature.to(torch.float16).cpu().numpy()
    else:
        feature = np.asarray(feature, dtype=np.float16)

    index = None
    if index_path is not None and Path(index_path).exists():
        index = _torch_load_first(index_path)
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        index = np.asarray(index, dtype=np.int64).reshape(-1)
    return sanitize_feature(feature), index


def load_npy_feature(feature_path, index_path=None):
    feature = sanitize_feature(np.load(feature_path))
    index = None
    if index_path is not None and Path(index_path).exists():
        index = np.asarray(np.load(index_path), dtype=np.int64).reshape(-1)
    return feature, index


def resolve_occamlgs_feature_paths(feature_root, scene_name):
    if feature_root is None:
        return None, None, None

    feature_dir = Path(feature_root) / scene_name
    candidates = [
        ("npy", feature_dir / "lang_feat.npy", feature_dir / "lang_feat_index.npy"),
        ("pth", feature_dir / "lang_feat.pth", feature_dir / "lang_feat_index.pth"),
        ("legacy_pth", feature_dir / "langfeat.pth", feature_dir / "langfeat_index.pth"),
    ]
    for output_format, feature_path, index_path in candidates:
        if feature_path.exists():
            return feature_path, index_path if index_path.exists() else None, output_format
    return None, None, None


def load_occamlgs_feature(feature_root, scene_name):
    feature_path, index_path, output_format = resolve_occamlgs_feature_paths(
        feature_root, scene_name
    )
    if feature_path is None:
        raise FileNotFoundError(
            f"No OccamLGS language feature found for scene '{scene_name}' under {feature_root}. "
            "Expected lang_feat.npy, lang_feat.pth, or legacy langfeat.pth."
        )
    if output_format == "npy":
        return load_npy_feature(feature_path, index_path)
    return load_torch_feature(feature_path, index_path)


def resolve_ludvig_feature_paths(feature_root, scene_name, model_dir, feature_name):
    if feature_root is None:
        return None, None

    feature_dir = Path(feature_root) / scene_name / model_dir
    feature_candidates = [
        feature_dir / f"{feature_name}.npy",
        feature_dir / "features.npy",
    ]
    for feature_path in feature_candidates:
        if not feature_path.exists():
            continue

        index_candidates = [
            feature_path.with_name(f"{feature_name}_index.npy"),
            feature_path.with_name("features_index.npy"),
        ]
        index_path = next((path for path in index_candidates if path.exists()), None)
        return feature_path, index_path
    return None, None


def save_compact_feature(feature, source_index, within_mask, save_path, feature_name, write_valid_mask=False):
    local_index, compact_feature = compact_feature_for_mask(feature, source_index, within_mask)
    save_path = Path(save_path)
    np.save(save_path / f"{feature_name}.npy", compact_feature)
    np.save(save_path / f"{feature_name}_index.npy", local_index)

    if write_valid_mask:
        valid_mask = np.zeros(len(np.asarray(within_mask)), dtype=bool)
        valid_mask[local_index] = True
        np.save(save_path / "valid_feat_mask.npy", valid_mask)
    return local_index, compact_feature


def compact_feature_for_mask(feature, source_index, within_mask):
    feature = sanitize_feature(feature)
    within_mask = np.asarray(within_mask, dtype=np.int64).reshape(-1)
    if within_mask.size == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, feature.shape[1]), dtype=feature.dtype),
        )

    if source_index is None:
        max_index = int(within_mask.max())
        if max_index >= len(feature):
            raise ValueError(
                f"Dense feature has {len(feature)} rows, but within_mask references row {max_index}."
            )
        feature_within = feature[within_mask]
        row_valid = np.any(feature_within != 0.0, axis=1)
        local_index = np.flatnonzero(row_valid).astype(np.int32)
        return local_index, feature_within[row_valid]

    source_index = np.asarray(source_index, dtype=np.int64).reshape(-1)
    if len(source_index) != len(feature):
        raise ValueError(
            f"Compact feature/index mismatch: {len(feature)} feature rows vs {len(source_index)} indices."
        )

    order = np.argsort(within_mask)
    sorted_within = within_mask[order]
    pos = np.searchsorted(sorted_within, source_index)
    in_range = pos < len(sorted_within)
    match = np.zeros(len(source_index), dtype=bool)
    match[in_range] = sorted_within[pos[in_range]] == source_index[in_range]

    if not match.any():
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, feature.shape[1]), dtype=feature.dtype),
        )

    local_index = order[pos[match]]
    compact_feature = feature[match]
    row_valid = np.any(compact_feature != 0.0, axis=1)
    return local_index[row_valid].astype(np.int32), compact_feature[row_valid]


def sanitize_feature(feature):
    if feature.dtype != np.float16:
        feature = feature.astype(np.float16)
    np.nan_to_num(feature, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feature


def _torch_load_first(path):
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if isinstance(data, (tuple, list)):
        data = data[0]
    return data
