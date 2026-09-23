import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import os
from plyfile import PlyData
from sklearn.neighbors import KDTree
import torch
import open3d as o3d
from pointcept.utils.misc import cancel_and_terminate_pool
from pointcept.datasets.preprocessing.feature_utils import (
    compact_feature_for_mask,
    copy_base_npy_scene,
    find_npy_scene_dir,
    load_occamlgs_feature,
    load_npy_feature,
    resolve_ludvig_feature_paths,
    resolve_occamlgs_feature_paths,
    save_compact_feature,
    write_compact_teacher_features,
)

################################################################################
# I/O Utilities
################################################################################


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".npy"]:
            return cls._read_npy(file_path)
        elif file_extension in [".h5"]:
            return cls._read_h5(file_path)
        elif file_extension in [".txt"]:
            return cls._read_txt(file_path)
        elif file_extension in [".ply"]:
            return cls._read_ply(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, "r")
        return f["data"][()]

    @classmethod
    def _read_ply(cls, file_path):
        return PlyData.read(file_path)


################################################################################
# Gaussian Reading
################################################################################


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_gaussian_attribute(
    vertex, attribute=["coord", "opacity", "scale", "quat", "color"]
):
    """
    Reads a 'vertex' structure from a PlyData file and returns a dictionary
    with 'coord', 'opacity', 'scale', 'quat', 'color' (all optional).
    """
    data = {}

    # Coordinates (xyz)
    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    data["coord"] = np.stack((x, y, z), axis=-1)  # [N, 3]

    # Opacity
    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32)
        opacity = np_sigmoid(opacity)  # range (0,1)
        data["opacity"] = opacity

    # Scale & Quaternion
    if "scale" in attribute and ("quat" in attribute or "euler" in attribute):
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data["coord"].shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)
        scales = np.exp(scales)  # exponentiate to get actual scale
        data["scale"] = scales

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data["coord"].shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        # Normalize the quaternion
        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # Enforce positive real part
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]
        data["quat"] = rots

    # Color (from spherical harmonics DC term) or direct color
    if "sh" in attribute or "color" in attribute:
        # DC terms
        features_dc = np.zeros((data["coord"].shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

        feature_pc = features_dc.reshape(-1, 3)
        # Move from SH DC to approximate color
        C0 = 0.28209479177387814  # Spherical Harmonics Y00
        feature_pc = (feature_pc * C0).astype(np.float32) + 0.5
        feature_pc = np.clip(feature_pc, 0, 1)
        # Store 0-255
        data["color"] = (feature_pc * 255).astype(np.uint8)

    return data


################################################################################
# Utility: find folder that ends with a certain suffix
################################################################################


def find_folder_with_suffix(root_dir, suffix):
    """
    Return the first folder under `root_dir` whose name ends with `suffix`.
    """
    root = Path(root_dir)
    matching_folders = [
        folder
        for folder in root.rglob("*")
        if folder.is_dir() and folder.name.endswith(suffix)
    ]
    if len(matching_folders) == 0:
        raise FileNotFoundError(f"No folder with suffix {suffix} found in {root_dir}")
    return matching_folders


################################################################################
# Main Parsing Logic
################################################################################


def parse_scene(
    scene_name,
    split,
    gs_root,
    pc_root,
    output_root,
    feat_root=None,
    skip_feat=True,
    remove_feat=False,
    debug=False,
    bbox_pruning=False,
    bbox_enlargement=0.25,
    dino_root=None,
    pe_root=None,
    nn_dist_thre=0.35,
    opacity_thre=0.1,
):
    """
    scene_name: e.g., '17DRP5sb8fy_00'
    split: 'train', 'val', or 'test'
    gs_root: Path to your 3D Gaussians. Scenes are all in a single folder structure.
    pc_root: Path to point cloud subfolders, each subfolder has coord.npy, segment.npy, etc.
    output_root: Where we save the per-scene Gaussian data
    debug: If True, skip actually writing data
    """
    if debug:
        print("===================")
        print("DEBUG MODE, TURN OFF TO SAVE DATA")

    print(f"Parsing scene {scene_name} in {split} split")

    npy_scene_dir = find_npy_scene_dir(gs_root, scene_name, split)
    if npy_scene_dir is not None:
        if bbox_pruning:
            print(
                f"[{scene_name}] Found released NPY scene at {npy_scene_dir}; "
                "--bbox_pruning is ignored because the base scene is already preprocessed."
            )
        save_path = Path(output_root) / split / scene_name
        copied = copy_base_npy_scene(npy_scene_dir, save_path)
        coord_count = len(np.load(npy_scene_dir / "coord.npy", mmap_mode="r"))
        written = write_compact_teacher_features(
            scene_name,
            save_path,
            coord_count,
            feat_root=None if skip_feat else feat_root,
            dino_root=dino_root,
            pe_root=pe_root,
        )
        print(
            f"[{scene_name}] copied {len(copied)} base NPY assets from {npy_scene_dir}; "
            f"updated teacher features: {written if written else 'none'}."
        )
        return

    # 1) Find the folder in gs_root that ends with scene_name
    scene_path_candidates = find_folder_with_suffix(gs_root, scene_name)
    scene_path = scene_path_candidates[0]
    gs_path = scene_path / "ckpts" / "point_cloud_250000.ply"
    feat_path, feat_index_path, _ = resolve_occamlgs_feature_paths(feat_root, scene_name)
    dino_path, dino_index_path = resolve_ludvig_feature_paths(
        dino_root, scene_name, "dinov3", "dino_feat"
    )
    pe_path, pe_index_path = resolve_ludvig_feature_paths(
        pe_root, scene_name, "pe_spatial", "pe_feat"
    )

    # 2) Load the GS file
    try:
        gs = IO.get(gs_path)
    except Exception as e:
        print(f"Error loading {gs_path}: {e}")
        return  # skip this scene

    # 3) Parse the Gaussians
    vertex = gs["vertex"]
    gs_data = read_gaussian_attribute(
        vertex, attribute=["coord", "opacity", "scale", "quat", "color"]
    )

    coord = gs_data["coord"].astype(np.float16)
    color = gs_data["color"].astype(np.uint8)
    opacity = gs_data["opacity"].astype(np.float16)
    scale = gs_data["scale"].astype(np.float16)
    quat = gs_data["quat"].astype(np.float16)

    # 4) Load the PC from pc_root for nearest neighbor labeling
    #    We'll look for pc_root/ split / scene_name
    scene_pc_dir = Path(pc_root) / split / scene_name
    pc_coord_path = scene_pc_dir / "coord.npy"
    pc_segment_path = scene_pc_dir / "segment_nyu_160.npy"
    pc_normal_path = scene_pc_dir / "normal.npy"
    pc_segment_nyu_path = scene_pc_dir / "segment_nyu_160.npy"

    if not pc_coord_path.exists() or not pc_segment_path.exists():
        print(
            f"Point cloud or segment file not found for scene {scene_name}. Skipping."
        )
        return

    gs_feat_np = None
    gs_feat_index = None
    if feat_path and not skip_feat:
        gs_feat_np, gs_feat_index = load_occamlgs_feature(feat_root, scene_name)
        if gs_feat_index is None:
            assert len(coord) == len(gs_feat_np), (
                f"coord and gs_feat_np not match {len(coord)} and {len(gs_feat_np)} in {scene_name}"
            )
        else:
            assert len(gs_feat_index) == len(gs_feat_np), (
                f"lang_feat and lang_feat_index not match {len(gs_feat_np)} and {len(gs_feat_index)} in {scene_name}"
            )

    pc_coord = np.load(pc_coord_path)  # (N, 3)
    pc_segment = np.load(pc_segment_path)  # (N,) or (N,1)
    pc_normal = np.load(pc_normal_path) if pc_normal_path.exists() else None  # (N, 3)
    # handle shape
    if pc_segment.ndim == 2 and pc_segment.shape[1] == 1:
        pc_segment = pc_segment.squeeze(1)

    # If segment_nyu_160.npy exists, read it
    pc_segment_nyu = None
    # if pc_segment_nyu_path.exists():
    #     pc_segment_nyu = np.load(pc_segment_nyu_path)
    #     if pc_segment_nyu.ndim == 2 and pc_segment_nyu.shape[1] == 1:
    #         pc_segment_nyu = pc_segment_nyu.squeeze(1)

    # 5) Optional bounding-box-based pruning
    if bbox_pruning:
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pc_coord)
        oriented_bbox = pc_o3d.get_minimal_oriented_bounding_box()
        enlargement = float(bbox_enlargement)
        new_extent = np.asarray(oriented_bbox.extent) + 2 * enlargement
        oriented_bbox.extent = new_extent

        gs_o3d = o3d.geometry.PointCloud()
        gs_o3d.points = o3d.utility.Vector3dVector(coord.astype(np.float32))

        within_mask = oriented_bbox.get_point_indices_within_bounding_box(
            gs_o3d.points
        )
        print(f"Pruned {len(coord) - len(within_mask)} gaussians by bounding box.")
        coord = coord[within_mask]
        color = color[within_mask]
        opacity = opacity[within_mask]
        scale = scale[within_mask]
        quat = quat[within_mask]
    else:
        within_mask = None

    # 6) Nearest Neighbor from point clouds to get the semantic label, and normal for each Gaussian
    tree = KDTree(pc_coord)
    _, indices = tree.query(coord, k=1)
    indices = indices[:, 0]
    gs_segment = pc_segment[indices]
    if gs_segment.ndim > 1:
        gs_segment = gs_segment.squeeze(1)
    nn_dist = np.linalg.norm(pc_coord[indices] - coord, axis=1)
    print(
        "gs to pc NN distance: mean",
        float(nn_dist.mean()),
        "max",
        float(nn_dist.max()),
    )
    exceed_mask = nn_dist > nn_dist_thre
    if exceed_mask.any():
        print(
            f"[{scene_name}]: {exceed_mask.sum()} gaussians exceed nn_dist_thre={nn_dist_thre}; "
            "setting labels to -1."
        )
    low_opacity_mask = opacity < opacity_thre
    if low_opacity_mask.any():
        print(
            f"[{scene_name}]: {low_opacity_mask.sum()} gaussians have opacity < {opacity_thre}; "
            "setting labels to -1."
        )
    invalid_mask = exceed_mask | low_opacity_mask

    # If we also have NYU 160 class labels
    gs_segment_nyu = None
    gs_normal = None
    if pc_segment_nyu is not None:
        gs_segment_nyu = pc_segment_nyu[indices]
        if gs_segment_nyu.ndim > 1:
            gs_segment_nyu = gs_segment_nyu.squeeze(1)
    if pc_normal is not None:
        gs_normal = pc_normal[indices]
        if gs_normal.ndim > 2 and gs_normal.shape[1] == 1:
            gs_normal = gs_normal[:, 0]

    gs_segment = gs_segment.astype(np.int32)
    if gs_segment_nyu is not None:
        gs_segment_nyu = gs_segment_nyu.astype(np.int32)
    if invalid_mask.any():
        gs_segment[invalid_mask] = -1
        if gs_segment_nyu is not None:
            gs_segment_nyu[invalid_mask] = -1
    # 7) Save results
    save_path = Path(output_root) / split / scene_name
    save_path.mkdir(parents=True, exist_ok=True)

    if not skip_feat and gs_feat_np is not None:
        feature_mask = (
            np.arange(len(coord), dtype=np.int64) if within_mask is None else within_mask
        )
        valid_indices, lang_feat = compact_feature_for_mask(
            gs_feat_np, gs_feat_index, feature_mask
        )
        valid_feat_mask = np.zeros(len(feature_mask), dtype=bool)
        valid_feat_mask[valid_indices] = True
        np.save(save_path / "valid_feat_mask.npy", valid_feat_mask)
        np.save(save_path / "lang_feat_index.npy", valid_indices)
        np.save(save_path / "lang_feat.npy", lang_feat)
        print(f"[{scene_name}] saved lang_feat with {len(valid_indices)}/{len(feature_mask)} valid rows.")

    def scrub_inplace_chunks(A, rows_per_chunk=500_000):
        # A is a writable array (e.g., already masked down with within_mask)
        n = A.shape[0]
        for i in range(0, n, rows_per_chunk):
            sub = A[i:i+rows_per_chunk]
            np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def rows_not_all_zero(A, rows_per_chunk=500_000):
        """
        Return a boolean mask where True means the row has at least one non-zero entry.
        Chunked to keep memory usage modest.
        """
        n = A.shape[0]
        out = np.empty(n, dtype=bool)
        for i in range(0, n, rows_per_chunk):
            j = min(n, i + rows_per_chunk)
            sub = A[i:j]
            # Row is kept if any element is non-zero
            out[i:j] = np.any(sub != 0, axis=1)
        return out

    # DINO features (optional)
    if dino_path is not None and dino_path.exists():
        dino_feat, dino_index = load_npy_feature(dino_path, dino_index_path)
        feature_mask = (
            np.arange(len(coord), dtype=np.int64) if within_mask is None else within_mask
        )
        dino_indices, _ = save_compact_feature(
            dino_feat, dino_index, feature_mask, save_path, "dino_feat"
        )
        print(f"[{scene_name}] saved dino_feat with {len(dino_indices)}/{len(feature_mask)} valid rows.")

    if pe_path is not None and pe_path.exists():
        pe_feat, pe_index = load_npy_feature(pe_path, pe_index_path)
        feature_mask = (
            np.arange(len(coord), dtype=np.int64) if within_mask is None else within_mask
        )
        pe_indices, _ = save_compact_feature(
            pe_feat, pe_index, feature_mask, save_path, "pe_feat"
        )
        print(f"[{scene_name}] saved pe_feat with {len(pe_indices)}/{len(feature_mask)} valid rows.")

    if not debug:
        np.save(save_path / "coord.npy", coord)
        np.save(save_path / "color.npy", color)
        np.save(save_path / "opacity.npy", opacity)
        np.save(save_path / "scale.npy", scale)
        np.save(save_path / "quat.npy", quat)
        # Save nearest-neighbor semantic data
        np.save(save_path / "segment_nyu_160.npy", gs_segment.astype(np.int32))
        if gs_segment_nyu is not None:
            np.save(save_path / "segment_nyu_160.npy", gs_segment_nyu.astype(np.int32))
        if gs_normal is not None:
            np.save(save_path / "normal.npy", gs_normal.astype(np.float16))

        # Save pc_* only for test split
        if split == "test":
            np.save(save_path / "pc_coord.npy", pc_coord.astype(np.float16))
            np.save(save_path / "pc_segment_nyu_160.npy", pc_segment.astype(np.int32))
            if pc_segment_nyu is not None:
                np.save(
                    save_path / "pc_segment_nyu_160.npy", pc_segment_nyu.astype(np.int32)
                )
    else:
        print("Debug mode: not saving data for scene:", scene_name)

    if remove_feat:
        # Remove the feature file if it exists
        if feat_path and feat_path.exists():
            os.remove(feat_path)
            print(f"Removed feature file: {feat_path}")

    print(f"Scene {scene_name} processed successfully!")


################################################################################
# Main script
################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gs_root",
        required=True,
        help="Released base NPY 3DGS root, or older raw Gaussian PLY root.",
    )
    parser.add_argument(
        "--pc_root",
        required=True,
        help="Path to the Matterport3D preprocessed point cloud dataset.",
    )
    parser.add_argument(
        "--feat_root",
        help="Path to language features.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val/test folders will be located.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, will not save data. For debugging.",
    )
    parser.add_argument(
        "--skip_feat", action="store_true", help="Skip feature processing."
    )
    parser.add_argument(
        "--remove_feat", action="store_true", help="Skip feature processing."
    )
    parser.add_argument(
        "--bbox_pruning",
        action="store_true",
        help="Enable pruning Gaussians to within an enlarged PC oriented bbox.",
    )
    parser.add_argument(
        "--bbox_enlargement",
        type=float,
        default=0.25,
        help="Enlargement margin added to each side when bbox pruning is enabled.",
    )
    parser.add_argument(
        "--nn_dist_thre",
        type=float,
        default=0.35,
        help="Ignore labels whose nearest-neighbor distance exceeds this threshold (meters).",
    )
    parser.add_argument(
        "--opacity_thre",
        type=float,
        default=0.05,
        help="Ignore labels whose opacity is below this threshold.",
    )
    parser.add_argument(
        "--dino_root",
        help="Path to LUDVIG DINOv3 root (expects <scene>/dinov3/dino_feat.npy + dino_feat_index.npy, or legacy features.npy).",
    )
    parser.add_argument(
        "--pe_root",
        help="Path to LUDVIG PE-Spatial root (expects <scene>/pe_spatial/pe_feat.npy + pe_feat_index.npy, or legacy features.npy).",
    )
    parser.add_argument(
        "--single_process",
        action="store_true",
        help="Disable parallel processing and run sequentially.",
    )
    config = parser.parse_args()

    config.gs_root = Path(config.gs_root)
    config.pc_root = Path(config.pc_root)
    config.feat_root = Path(config.feat_root) if config.feat_root else None
    config.dino_root = Path(config.dino_root) if config.dino_root else None
    config.pe_root = Path(config.pe_root) if config.pe_root else None
    config.output_root = Path(config.output_root)

    # Collect the scenes from train/val/test subfolders in pc_root
    def get_scenes_from_split(split_name):
        split_dir = config.pc_root / split_name
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist!")
        # Scenes are subfolders
        return sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    train_scenes = get_scenes_from_split("train")
    val_scenes = get_scenes_from_split("val")
    test_scenes = get_scenes_from_split("test")

    # Keep scenes that exist either in the released NPY layout
    # (<root>/<split>/<scene>) or in the older raw-GS layout (<root>/<scene>/ckpts).
    train_scenes = [
        scene
        for scene in train_scenes
        if find_npy_scene_dir(config.gs_root, scene, "train") is not None
        or (config.gs_root / scene).exists()
    ]
    val_scenes = [
        scene
        for scene in val_scenes
        if find_npy_scene_dir(config.gs_root, scene, "val") is not None
        or (config.gs_root / scene).exists()
    ]
    test_scenes = [
        scene
        for scene in test_scenes
        if find_npy_scene_dir(config.gs_root, scene, "test") is not None
        or (config.gs_root / scene).exists()
    ]
    data_list = train_scenes + val_scenes + test_scenes

    print("Num train scenes:", len(train_scenes))
    print("Num val scenes:", len(val_scenes))
    print("Num test scenes:", len(test_scenes))
    print("Total scenes:", len(data_list))

    # Combine them for easy iteration
    data_list = train_scenes + val_scenes + test_scenes
    split_list = ["train"] * len(train_scenes) + ["val"] * len(val_scenes) + ["test"] * len(test_scenes)

    # Parallel processing
    if config.single_process:
        # Sequential processing
        print("Running in sequential mode...")
        try:
            for scene_name, split_name in zip(data_list, split_list):
                parse_scene(
                    scene_name,
                    split_name,
                    config.gs_root,
                    config.pc_root,
                    config.output_root,
                    config.feat_root,
                    config.skip_feat,
                    config.remove_feat,
                    config.debug,
                    config.bbox_pruning,
                    config.bbox_enlargement,
                    config.dino_root,
                    config.pe_root,
                    config.nn_dist_thre,
                    config.opacity_thre,
                )
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping...")
            raise SystemExit(130)
        except Exception as e:
            print(f"[{scene_name}] Exception occurred: {e}", flush=True)
            raise
    else:
        print(f"Running with {config.num_workers} workers...")
        pool = None
        futures = []
        try:
            pool = ProcessPoolExecutor(max_workers=config.num_workers)
            for scene_name, split_name in zip(data_list, split_list):
                futures.append(
                    pool.submit(
                        parse_scene,
                        scene_name,
                        split_name,
                        config.gs_root,
                        config.pc_root,
                        config.output_root,
                        config.feat_root,
                        config.skip_feat,
                        config.remove_feat,
                        config.debug,
                        config.bbox_pruning,
                        config.bbox_enlargement,
                        config.dino_root,
                        config.pe_root,
                        config.nn_dist_thre,
                        config.opacity_thre,
                    )
                )
            # Wait for all futures to complete
            for future in futures:
                future.result()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, cleaning up...", flush=True)
            if pool is not None:
                cancel_and_terminate_pool(pool, futures)
            print("Cleanup complete, exiting.", flush=True)
            raise SystemExit(130)
        except Exception as e:
            print(f"[{scene_name}] Exception occurred: {e}", flush=True)
            if pool is not None:
                cancel_and_terminate_pool(pool, futures)
            raise

    print("Done preprocessing Matterport3D Gaussian data!")


if __name__ == "__main__":
    main()
