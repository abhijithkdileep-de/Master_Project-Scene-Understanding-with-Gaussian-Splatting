"""
Preprocessing Script for ScanNet++ 3DGS with language features,
This is comatible for both MCMC and fixed_xyz version of 3DGS.
"""

import argparse
import numpy as np
import pandas as pd
import open3d as o3d
import multiprocessing as mp
from multiprocessing import Manager
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat
from pathlib import Path

import h5py
import os
import torch
from plyfile import PlyData
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


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".npy"]:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in [".h5"]:
            return cls._read_h5(file_path)
        elif file_extension in [".txt"]:
            return cls._read_txt(file_path)
        elif file_extension in [".ply"]:
            return cls._read_ply(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

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


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_gaussian_attribute(
    vertex, attribute=["coord", "opacity", "scale", "quat", "color"]
):
    # print(vertex.data.dtype.names)
    # assert "coord" in attribute, "At least need xyz attribute" can free this one actually
    # record the attribute and the index to read it
    data = dict()

    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    # data = np.stack((x, y, z), axis=-1) # [n, 3]
    data["coord"] = np.stack((x, y, z), axis=-1)  # [n, 3]

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32)
        opacity = np_sigmoid(opacity)  # sigmoid activation
        data["opacity"] = opacity

    if "scale" in attribute and ("quat" in attribute or "euler" in attribute):
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data["coord"].shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)

        scales = np.exp(scales)  # scale activation
        data["scale"] = scales

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data["coord"].shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # always set the first to be positive
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]
        data["quat"] = rots

    if "sh" in attribute or "color" in attribute:
        # sphere homrincals to rgb
        features_dc = np.zeros((data["coord"].shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

        feature_pc = features_dc.reshape(-1, 3)
        if "color" in attribute:
            C0 = 0.28209479177387814
            feature_pc = (feature_pc * C0).astype(np.float32) + 0.5
            feature_pc = np.clip(feature_pc, 0, 1)
        # data = np.concatenate((data, feature_pc), axis=1)
        data["color"] = feature_pc * 255

    return data


def find_folder_with_suffix(root_dir, suffix):
    # Convert the root directory to a Path object
    root = Path(root_dir)

    # Search for directories whose names end with the given suffix
    matching_folders = [
        folder
        for folder in root.rglob("*")
        if folder.is_dir() and folder.name.endswith(suffix)
    ]
    assert len(matching_folders) > 0, (
        f"No folder with suffix {suffix} found in {root_dir}"
    )

    return matching_folders


def parse_scene(
    name,
    split,
    dataset_root,
    gs_root,
    pc_root,
    output_root,
    label_mapping,
    class2idx,
    ignore_index=-1,
    feat_root=None,
    feat_only=False,
    skip_feat=False,
    shared_skipped_scenes=None,
    debug=False,
    bbox_pruning=False,
    bbox_enlargement=0.25,
    dino_root=None,
    pe_root=None,
    nn_dist_thre=0.35,
    opacity_thre=0.1,
):
    if debug:
        print("===================")
        print("DEBUG MODE, TURN OFF TO SAVE DATA")
    print(f"Parsing scene {name} in {split} split")
    dataset_root = Path(dataset_root)
    gs_root = Path(gs_root)
    pc_root = Path(pc_root)
    output_root = Path(output_root)

    npy_scene_dir = find_npy_scene_dir(gs_root, name, split)
    if npy_scene_dir is not None:
        if feat_root is None and not skip_feat:
            raise ValueError("feat_root is None but skip_feat is False")
        if bbox_pruning:
            print(
                f"[{name}] Found released NPY scene at {npy_scene_dir}; "
                "--bbox_pruning is ignored because the base scene is already preprocessed."
            )
        save_path = output_root / split / name
        copied = copy_base_npy_scene(npy_scene_dir, save_path)
        coord_count = len(np.load(npy_scene_dir / "coord.npy", mmap_mode="r"))
        written = write_compact_teacher_features(
            name,
            save_path,
            coord_count,
            feat_root=None if skip_feat else feat_root,
            dino_root=dino_root,
            pe_root=pe_root,
        )
        print(
            f"[{name}] copied {len(copied)} base NPY assets from {npy_scene_dir}; "
            f"updated teacher features: {written if written else 'none'}."
        )
        return

    scene_path = find_folder_with_suffix(gs_root, name)[0]
    print("scene_path", scene_path)
    data_path = dataset_root / "data" if split != "test" else dataset_root / "sem_test"
    mesh_path = data_path / str(name) / "scans" / "mesh_aligned_0.05.ply"

    gs_path = scene_path / "ckpts" / "point_cloud_30000.ply"
    pc_path = pc_root / split / name / "coord.npy"
    pc_instance = pc_root / split / name / "instance.npy"
    pc_semantic = pc_root / split / name / "segment.npy"
    pc_normal = pc_root / split / name / "normal.npy"
    feat_path, feat_index_path, _ = resolve_occamlgs_feature_paths(feat_root, name)
    dino_path, dino_index_path = resolve_ludvig_feature_paths(
        dino_root, name, "dinov3", "dino_feat"
    )
    pe_path, pe_index_path = resolve_ludvig_feature_paths(
        pe_root, name, "pe_spatial", "pe_feat"
    )
    if feat_path is None:
        if not skip_feat:
            raise ValueError("feat_root is None but skip_feat is False")
        else:
            print("Skipping feature processing...")

    try:
        gs = IO.get(gs_path)
    except Exception as e:
        print("Error in loading:")
        raise e

    ##### speeding up when doing feat_only
    if feat_only:
        gs_feat_np = None
        gs_feat_index = None
        source_indices = np.arange(gs["vertex"].count, dtype=np.int64)
        if feat_path is not None:
            gs_feat_np, gs_feat_index = load_occamlgs_feature(feat_root, name)
            valid_indices, lang_feat = compact_feature_for_mask(
                gs_feat_np, gs_feat_index, source_indices
            )
            valid_percent = len(valid_indices) / len(source_indices)
            if valid_percent < 0.5:
                print(
                    f"Valid language feature percent too low {valid_percent}, skipping scene {name}..."
                )
                shared_skipped_scenes.append(name)
                return
        save_path = output_root / split / name
        save_path.mkdir(parents=True, exist_ok=True)
        if gs_feat_np is not None:
            valid_mask = np.zeros(len(source_indices), dtype=bool)
            valid_mask[valid_indices] = True
            np.save(save_path / "valid_feat_mask.npy", valid_mask)
            np.save(save_path / "lang_feat.npy", lang_feat)
            np.save(save_path / "lang_feat_index.npy", valid_indices)
        if dino_path is not None and os.path.exists(dino_path):
            dino_feat, dino_index = load_npy_feature(dino_path, dino_index_path)
            save_compact_feature(dino_feat, dino_index, source_indices, save_path, "dino_feat")
        if pe_path is not None and os.path.exists(pe_path):
            pe_feat, pe_index = load_npy_feature(pe_path, pe_index_path)
            save_compact_feature(pe_feat, pe_index, source_indices, save_path, "pe_feat")
        print(f"Saved features to {save_path}")
        return

    vertex = gs["vertex"]
    gs_data = read_gaussian_attribute(
        vertex, attribute=["coord", "opacity", "scale", "quat", "color"]
    )

    # create mesh from coord in gs_data
    # mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    # mesh.compute_vertex_normals(normalized=True)
    # normal = np.array(mesh.vertex_normals).astype(np.float32)
    normal = np.load(pc_normal)

    # get gaussian instance and semantic labels from nearest pointcloud
    pc_coord = np.load(pc_path)
    if split != "test":
        pc_instance = np.load(pc_instance)
        pc_semantic = np.load(pc_semantic)
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_coord)

    coord = gs_data["coord"].astype(np.float16)
    color = gs_data["color"].astype(np.uint8)
    opacity = gs_data["opacity"].astype(np.float16)
    scale = gs_data["scale"].astype(np.float16)
    quat = gs_data["quat"].astype(np.float16)

    gs_o3d = o3d.geometry.PointCloud()
    gs_o3d.points = o3d.utility.Vector3dVector(coord)

    gs_feat_np = None
    gs_feat_index = None
    if feat_path is not None and not skip_feat:
        gs_feat_np, gs_feat_index = load_occamlgs_feature(feat_root, name)
        if gs_feat_index is None:
            assert len(coord) == len(gs_feat_np), (
                f"coord and gs_feat_np not match {len(coord)} and {len(gs_feat_np)}"
            )
        else:
            assert len(gs_feat_index) == len(gs_feat_np), (
                f"lang_feat and lang_feat_index not match {len(gs_feat_np)} and {len(gs_feat_index)}"
            )
    # DINO features (optional)
    dino_feat_np = None
    dino_feat_index = None
    if dino_path is not None and os.path.exists(dino_path):
        dino_feat_np, dino_feat_index = load_npy_feature(dino_path, dino_index_path)
        if dino_feat_index is None:
            assert len(coord) == len(dino_feat_np), (
                f"coord and dino_feat_np not match: coord {len(coord)} and dino_feat_np {len(dino_feat_np)}"
            )
    pe_feat_np = None
    pe_feat_index = None
    if pe_path is not None and os.path.exists(pe_path):
        pe_feat_np, pe_feat_index = load_npy_feature(pe_path, pe_index_path)
        if pe_feat_index is None:
            assert len(coord) == len(pe_feat_np), (
                f"coord and pe_feat_np not match: coord {len(coord)} and pe_feat_np {len(pe_feat_np)}"
            )

    # BBox pruning (optional)
    if bbox_pruning:
        oriented_bbox = pc_o3d.get_minimal_oriented_bounding_box()
        enlargement = float(bbox_enlargement)
        new_extent = np.asarray(oriented_bbox.extent) + 2 * enlargement
        oriented_bbox.extent = new_extent
        within_mask = oriented_bbox.get_point_indices_within_bounding_box(
            gs_o3d.points
        )
        print(
            "Pruned", len(coord) - len(within_mask), "gaussians by init bounding box."
        )
    else:
        within_mask = np.arange(len(coord))

    coord = coord[within_mask]
    color = color[within_mask]
    opacity = opacity[within_mask]
    scale = scale[within_mask]
    quat = quat[within_mask]

    save_path = output_root / split / name
    save_path.mkdir(parents=True, exist_ok=True)

    if gs_feat_np is not None:
        valid_indices, lang_feat = compact_feature_for_mask(
            gs_feat_np, gs_feat_index, within_mask
        )
        valid_percent = len(valid_indices) / len(within_mask)
        if not valid_percent < 0.5:
            valid_feat_mask = np.zeros(len(within_mask), dtype=bool)
            valid_feat_mask[valid_indices] = True
            np.save(save_path / "valid_feat_mask.npy", valid_feat_mask)
            np.save(save_path / "lang_feat.npy", lang_feat)
            np.save(save_path / "lang_feat_index.npy", valid_indices)
        else:
            print(
                f"Valid language feature percent too low {valid_percent}, skipping scene {name}..."
            )
            shared_skipped_scenes.append(name)

    # Save DINO features if available
    if dino_feat_np is not None:
        dino_indices, _ = save_compact_feature(
            dino_feat_np, dino_feat_index, within_mask, save_path, "dino_feat"
        )
        print(f"[{name}] saved dino_feat with {len(dino_indices)}/{len(within_mask)} valid rows.")

    if pe_feat_np is not None:
        pe_indices, _ = save_compact_feature(
            pe_feat_np, pe_feat_index, within_mask, save_path, "pe_feat"
        )
        print(f"[{name}] saved pe_feat with {len(pe_indices)}/{len(within_mask)} valid rows.")

    # get the nearest pointcloud points for each gaussian point
    from sklearn.neighbors import KDTree

    tree = KDTree(pc_coord)
    _, indices = tree.query(coord, k=1)
    indices = indices[:, 0]
    gs_normal = normal[indices]
    nn_dist = np.linalg.norm(pc_coord[indices] - coord, axis=1)
    print(
        "gs to pc NN distance: mean",
        float(nn_dist.mean()),
        "max",
        float(nn_dist.max()),
    )
    dist_exceed_mask = nn_dist > nn_dist_thre
    if dist_exceed_mask.any():
        print(
            f"{dist_exceed_mask.sum()} gaussians exceed nn_dist_thre={nn_dist_thre}; "
            f"setting labels to ignore_index={ignore_index}."
        )
    low_opacity_mask = opacity < opacity_thre
    if low_opacity_mask.any():
        print(
            f"{low_opacity_mask.sum()} gaussians have opacity < {opacity_thre}; "
            f"setting labels to ignore_index={ignore_index}."
        )
    invalid_mask = dist_exceed_mask | low_opacity_mask

    if not debug:
        np.save(save_path / "coord.npy", coord)
        np.save(save_path / "color.npy", color)
        np.save(save_path / "normal.npy", gs_normal.astype(np.float16))
        np.save(save_path / "opacity.npy", opacity)
        np.save(save_path / "scale.npy", scale)
        np.save(save_path / "quat.npy", quat)

    if split == "test":
        print(f"Saved scene data to {save_path}")
        return

    gs_instance = pc_instance[indices][:, 0].astype(np.int32)
    gs_semantic = pc_semantic[indices][:, 0].astype(np.int32)
    gs_instance[invalid_mask] = ignore_index
    gs_semantic[invalid_mask] = ignore_index

    np.save(save_path / "instance.npy", gs_instance)
    np.save(save_path / "segment.npy", gs_semantic)
    # Also copy point cloud-level data for validation split only
    if split == "val":
        np.save(save_path / "pc_coord.npy", pc_coord.astype(np.float16))
        np.save(save_path / "pc_instance.npy", pc_instance.astype(np.int32))
        np.save(save_path / "pc_segment.npy", pc_semantic.astype(np.int32))

    print(f"Saved scene data to {save_path}")


def filter_map_classes(mapping, count_thresh, count_type, mapping_type):
    if count_thresh > 0 and count_type in mapping.columns:
        mapping = mapping[mapping[count_type] >= count_thresh]
    if mapping_type == "semantic":
        map_key = "semantic_map_to"
    elif mapping_type == "instance":
        map_key = "instance_map_to"
    else:
        raise NotImplementedError
    # create a dict with classes to be mapped
    # classes that don't have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]

        # map to None or some other label -> don't add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> don't use this class
                if map_target == "None":
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
        except TypeError:
            # nan values -> no mapping, keep label as is
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return map_dict


if __name__ == "__main__":
    manager = Manager()
    shared_skipped_scenes = manager.list()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet++ dataset containing data/metadata/splits.",
    )
    parser.add_argument(
        "--gs_root",
        required=True,
        help="Released base NPY 3DGS root, or older raw Gaussian PLY root.",
    )
    parser.add_argument(
        "--pc_root",
        required=True,
        help="Path to the ScanNet++ Point Cloud dataset.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val/test folders will be located.",
    )
    parser.add_argument(
        "--ignore_index",
        default=-1,
        type=int,
        help="Default ignore index.",
    )
    parser.add_argument(
        "--nn_dist_thre",
        default=0.35,
        type=float,
        help="Ignore labels whose nearest-neighbor distance exceeds this threshold (meters).",
    )
    parser.add_argument(
        "--opacity_thre",
        default=0.05,
        type=float,
        help="Ignore labels whose opacity is below this threshold.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--feat_root",
        help="Path to language features.",
    )
    parser.add_argument(
        "--feat_only", action="store_true", help="Only process features."
    )
    parser.add_argument(
        "--skip_feat", action="store_true", help="Skip feature processing."
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
        help="Enlargement margin (in meters) added to each side when bbox pruning is enabled.",
    )
    parser.add_argument(
        "--dino_root",
        help="Path to LUDVIG DINOv3 root (expects <scene>/dinov3/dino_feat.npy + dino_feat_index.npy, or legacy features.npy).",
    )
    parser.add_argument(
        "--pe_root",
        help="Path to LUDVIG PE-Spatial root (expects <scene>/pe_spatial/pe_feat.npy + pe_feat_index.npy, or legacy features.npy).",
    )
    parser.add_argument("--scenes_list", help="Specify a list of scenes to process.")
    config = parser.parse_args()

    print("Loading meta data...")
    config.dataset_root = Path(config.dataset_root)
    config.output_root = Path(config.output_root)
    config.feat_root = Path(config.feat_root) if config.feat_root else None
    config.dino_root = Path(config.dino_root) if config.dino_root else None
    config.pe_root = Path(config.pe_root) if config.pe_root else None

    train_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_train.txt",
        dtype=str,
    )
    print("Num samples in training split:", len(train_list))

    val_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_val.txt",
        dtype=str,
    )
    print("Num samples in validation split:", len(val_list))

    test_list = np.loadtxt(
        config.dataset_root / "splits" / "sem_test.txt",
        dtype=str,
    )
    print("Num samples in testing split:", len(test_list))

    if config.scenes_list:
        target_scenes = np.loadtxt(config.scenes_list, dtype=str)
        train_list = np.intersect1d(train_list, target_scenes)
        val_list = np.intersect1d(val_list, target_scenes)
        test_list = np.intersect1d(test_list, target_scenes)

    data_list = np.concatenate([train_list, val_list, test_list])
    # data_list = np.concatenate([test_list])
    print("Total number of scenes to process:", len(data_list))
    split_list = np.concatenate(
        [
            np.full_like(train_list, "train"),
            np.full_like(val_list, "val"),
            np.full_like(test_list, "test"),
        ]
    )

    # Parsing label information and mapping
    segment_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    print("Num classes in segment class list:", len(segment_class_names))

    instance_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100_instance.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    print("Num classes in instance class list:", len(instance_class_names))

    label_mapping = pd.read_csv(
        config.dataset_root / "metadata" / "semantic_benchmark" / "map_benchmark.csv"
    )
    label_mapping = filter_map_classes(
        label_mapping, count_thresh=0, count_type="count", mapping_type="semantic"
    )
    class2idx = {
        class_name: idx for (idx, class_name) in enumerate(segment_class_names)
    }

    tasks = []
    future_to_scene = {}
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
        for args in zip(
            data_list,
            split_list,
            repeat(config.dataset_root),
            repeat(config.gs_root),
            repeat(config.pc_root),
            repeat(config.output_root),
            repeat(label_mapping),
            repeat(class2idx),
            repeat(config.ignore_index),
            repeat(config.feat_root),
            repeat(config.feat_only),
            repeat(config.skip_feat),
            repeat(shared_skipped_scenes),
            repeat(False),  # if debug or not
            repeat(config.bbox_pruning),
            repeat(config.bbox_enlargement),
            repeat(config.dino_root),
            repeat(config.pe_root),
            repeat(config.nn_dist_thre),
            repeat(config.opacity_thre),
        ):
            future = pool.submit(parse_scene, *args)
            tasks.append(future)
            future_to_scene[future] = args[0]

        try:
            for future in as_completed(tasks):
                try:
                    result = future.result()
                except Exception as e:
                    scene_info = future_to_scene.get(future, "unknown")
                    with open("failed_scenes.txt", "a") as f:
                        f.write(f"Error processing scene {scene_info}: {e}\n")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, cleaning up...", flush=True)
            cancel_and_terminate_pool(pool, tasks)
            raise SystemExit(130)
    print(f"Skipped the following scenes for lang_feat: {shared_skipped_scenes}")
    print("Preprocessing done!")
