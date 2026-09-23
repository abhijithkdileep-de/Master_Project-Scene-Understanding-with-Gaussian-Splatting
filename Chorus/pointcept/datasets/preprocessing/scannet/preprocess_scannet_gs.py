import warnings
import os
import argparse
import glob
import json
import plyfile
import torch
import numpy as np
import pandas as pd
import open3d as o3d
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from sklearn.neighbors import KDTree
from meta_data.scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_20
from pointcept.utils.misc import cancel_and_terminate_pool
from pointcept.datasets.preprocessing.feature_utils import (
    copy_base_npy_scene,
    find_npy_scene_dir,
    load_occamlgs_feature,
    load_npy_feature,
    resolve_ludvig_feature_paths,
    resolve_occamlgs_feature_paths,
    save_compact_feature,
    write_compact_teacher_features,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Load external constants

CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDS200 = VALID_CLASS_IDS_200
CLASS_IDS20 = VALID_CLASS_IDS_20
IGNORE_INDEX = -1

###############################################################################
# 1) Utility Functions
###############################################################################

def read_plymesh(filepath):
    """
    Read the standard ScanNet mesh (with vertices and faces)
    and return as (vertices, faces). Returns (None, None) if empty.
    """
    with open(filepath, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
        return vertices, faces
    return None, None


def face_normal(vertex_coords, face):
    """
    Compute face normals + face areas for the given mesh.
    Returns (nf, area) for each face.
    """
    v01 = vertex_coords[face[:, 1]] - vertex_coords[face[:, 0]]
    v02 = vertex_coords[face[:, 2]] - vertex_coords[face[:, 0]]
    vec = np.cross(v01, v02)  # [F, 3]
    area = 0.5 * np.sqrt(np.sum(vec**2, axis=1, keepdims=True))
    length = np.maximum(1e-8, np.sqrt(np.sum(vec**2, axis=1, keepdims=True)))
    nf = vec / length
    return nf, area


def vertex_normal(vertex_coords, face):
    """
    Compute per-vertex normals by accumulating area-weighted face normals.
    """
    nf, area = face_normal(vertex_coords, face)
    nf_area = nf * area
    nv = np.zeros_like(vertex_coords)
    for i in range(face.shape[0]):
        inds = face[i]
        nv[inds[0]] += nf_area[i]
        nv[inds[1]] += nf_area[i]
        nv[inds[2]] += nf_area[i]
    lengths = np.maximum(1e-8, np.sqrt(np.sum(nv**2, axis=1, keepdims=True)))
    nv = nv / lengths
    return nv


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_gaussian_ply(filepath):
    """
    Reads a Gaussian Splat .ply with fields such as:
      x, y, z, opacity, scale_0, scale_1, ..., rot_0, rot_1, ..., f_dc_0, f_dc_1, ...
    Returns a dict with keys: coord, color, opacity, scale, quat.
    """
    with open(filepath, "rb") as f:
        ply_data = plyfile.PlyData.read(f)
    vertex = ply_data["vertex"]
    N = vertex.count

    # Basic coordinates
    coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(
        np.float32
    )

    # Opacity (using a sigmoid)
    if "opacity" in vertex.data.dtype.names:
        opacity_raw = vertex["opacity"].astype(np.float32)
        opacity = np_sigmoid(opacity_raw)
    else:
        opacity = np.ones(N, dtype=np.float32)

    # Scale values (exponentiated)
    scale_cols = [c for c in vertex.data.dtype.names if c.startswith("scale_")]
    scale_cols = sorted(scale_cols, key=lambda c: int(c.split("_")[-1]))
    scale = [np.exp(vertex[c]) for c in scale_cols]
    scale = (
        np.stack(scale, axis=-1).astype(np.float32)
        if scale
        else np.ones((N, 1), np.float32)
    )

    # Quaternion (rotation)
    rot_cols = [c for c in vertex.data.dtype.names if c.startswith("rot_")]
    rot_cols = sorted(rot_cols, key=lambda c: int(c.split("_")[-1]))
    quat = [vertex[c] for c in rot_cols]
    if quat:
        quat = np.stack(quat, axis=-1).astype(np.float32)
        length = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-9
        quat = quat / length
        sign_vector = np.sign(quat[:, 0])
        quat = quat * sign_vector[:, None]
    else:
        quat = np.ones((N, 4), dtype=np.float32)

    # Color from f_dc_0, f_dc_1, f_dc_2
    fdc_cols = [c for c in vertex.data.dtype.names if c.startswith("f_dc_")]
    fdc_cols = sorted(fdc_cols, key=lambda c: int(c.split("_")[-1]))
    if len(fdc_cols) >= 3:
        fdc_stack = [vertex[c] for c in fdc_cols]
        fdc_stack = np.stack(fdc_stack, axis=-1).astype(np.float32)
        C0 = 0.28209479177387814
        color = np.clip(fdc_stack * C0 + 0.5, 0, 1) * 255
        color = color.astype(np.uint8)
    else:
        color = np.full((N, 3), 128, dtype=np.uint8)

    return {
        "coord": coord,
        "color": color,
        "opacity": opacity,
        "scale": scale,
        "quat": quat,
    }


def point_indices_from_group(seg_indices, group, labels_pd):
    """
    For a group in the aggregation, map the raw label to 20- and 200-class IDs.
    """
    group_segments = np.array(group["segments"])
    label = group["label"]
    label_id20 = labels_pd[labels_pd["raw_category"] == label]["nyu40id"]
    label_id20 = int(label_id20.iloc[0]) if not label_id20.empty else 0
    label_id20 = (
        CLASS_IDS20.index(label_id20) if label_id20 in CLASS_IDS20 else IGNORE_INDEX
    )

    label_id200 = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id200 = int(label_id200.iloc[0]) if not label_id200.empty else 0
    label_id200 = (
        CLASS_IDS200.index(label_id200) if label_id200 in CLASS_IDS200 else IGNORE_INDEX
    )

    return group_segments, label_id20, label_id200


###############################################################################
# 2) Main Processing Function
###############################################################################


def handle_process(
    scene_path,
    output_path,
    labels_pd,
    train_scenes,
    val_scenes,
    gs_root,
    feat_root=None,
    feat_only=False,
    skip_feat=False,
    bbox_pruning=False,
    bbox_enlargement=0.25,
    dino_root=None,
    pe_root=None,
    nn_dist_thre=0.35,
    opacity_thre=0.1,
):
    """
    Process one scene.
      - For non-test splits: load the mesh and Gaussian splat data, compute normals, etc.
      - For test split: read the preprocessed point data (color.npy, coord.npy, normal.npy)
        from the given scene folder under preprocess_point/test/.
    """
    scene_id = os.path.basename(scene_path)
    print(f"Processing scene: {scene_id}")

    # dataset_root = Path(dataset_root)
    # pc_root = Path(pc_root)
    gs_root = Path(gs_root)
    feat_root = Path(feat_root) if feat_root else None
    dino_root = Path(dino_root) if dino_root else None
    pe_root = Path(pe_root) if pe_root else None

    if scene_id in train_scenes:
        output_path = os.path.join(output_path, "train", f"{scene_id}")
        split_name = "train"
    elif scene_id in val_scenes:
        output_path = os.path.join(output_path, "val", f"{scene_id}")
        split_name = "val"
    else:
        output_path = os.path.join(output_path, "test", f"{scene_id}")
        split_name = "test"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Standard mesh and, if applicable, segmentation/aggregation files from dataset_root.
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    segments_file = os.path.join(
        scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}"
    )
    aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")
    feat_path, feat_index_path, _ = resolve_occamlgs_feature_paths(feat_root, scene_id)
    if skip_feat or feat_path is None:
        print("Skipping feature processing...")

    npy_scene_dir = find_npy_scene_dir(gs_root, scene_id, split_name)
    if npy_scene_dir is not None:
        if bbox_pruning:
            print(
                f"[{scene_id}] Found released NPY scene at {npy_scene_dir}; "
                "--bbox_pruning is ignored because the base scene is already preprocessed."
            )
        copied = copy_base_npy_scene(npy_scene_dir, output_path)
        coord_count = len(np.load(npy_scene_dir / "coord.npy", mmap_mode="r"))
        written = write_compact_teacher_features(
            scene_id,
            output_path,
            coord_count,
            feat_root=None if skip_feat else feat_root,
            dino_root=dino_root,
            pe_root=pe_root,
        )
        print(
            f"[{scene_id}] copied {len(copied)} base NPY assets from {npy_scene_dir}; "
            f"updated teacher features: {written if written else 'none'}."
        )
        return

    vertices, faces = read_plymesh(mesh_path)
    if vertices is None:
        print(f"[WARN] No vertices found for {scene_id} at {mesh_path}")
        return

    mesh_coords = vertices[:, :3]
    mesh_normals = vertex_normal(mesh_coords, faces)

    # Oriented bbox prepared (used only if bbox_pruning is enabled)
    if bbox_pruning:
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(mesh_coords)
        oriented_bbox = pc_o3d.get_minimal_oriented_bounding_box()
        enlargement = float(bbox_enlargement)
        new_extent = np.asarray(oriented_bbox.extent) + 2 * enlargement
        oriented_bbox.extent = new_extent

    if split_name != "test":
        with open(segments_file) as f:
            segdata = json.load(f)
            seg_indices = np.array(segdata["segIndices"])
        with open(aggregations_file) as f:
            agg = json.load(f)
            seg_groups = agg["segGroups"]
    else:
        seg_indices = None
        seg_groups = []

    # Load Gaussian Splat data from gs_root
    gs_scene_dir = os.path.join(gs_root, scene_id, "ckpts")
    gs_candidates = glob.glob(os.path.join(gs_scene_dir, "*.ply"))
    print("GS scene directory:", gs_scene_dir)
    if len(gs_candidates) == 0:
        print(f"[WARN] No Gaussian .ply found for {scene_id}")
        return
    gs_path = gs_candidates[0]
    gs_data = read_gaussian_ply(gs_path)

    coord_gs = gs_data["coord"]
    color_gs = gs_data["color"]
    opacity_gs = gs_data["opacity"]
    scale_gs = gs_data["scale"]
    quat_gs = gs_data["quat"]
    N_gs = coord_gs.shape[0]

    gs_o3d = o3d.geometry.PointCloud()
    gs_o3d.points = o3d.utility.Vector3dVector(coord_gs)

    gs_feat_np = None
    gs_feat_index = None
    if feat_path and not skip_feat:
        gs_feat_np, gs_feat_index = load_occamlgs_feature(feat_root, scene_id)
        if gs_feat_index is None:
            assert len(coord_gs) == len(gs_feat_np), (
                f"coord and gs_feat_np not match {len(coord_gs)} and {len(gs_feat_np)} in {scene_id}"
            )
        else:
            assert len(gs_feat_index) == len(gs_feat_np), (
                f"lang_feat and lang_feat_index not match {len(gs_feat_np)} and {len(gs_feat_index)} in {scene_id}"
            )

    tree = KDTree(mesh_coords)
    _, nn_idx = tree.query(coord_gs, k=1)
    nn_idx = nn_idx[:, 0]
    nn_dist = np.linalg.norm(coord_gs - mesh_coords[nn_idx], axis=1)
    print(
        "Average distance of nearest pc neighbor:",
        float(nn_dist.mean()),
        "max:",
        float(nn_dist.max()),
    )
    dist_exceed_mask = nn_dist > nn_dist_thre
    if dist_exceed_mask.any():
        print(
            f"{dist_exceed_mask.sum()} gaussians exceed nn_dist_thre={nn_dist_thre}; "
            f"setting labels to IGNORE_INDEX={IGNORE_INDEX}."
        )
    low_opacity_mask = opacity_gs < opacity_thre
    if low_opacity_mask.any():
        print(
            f"{low_opacity_mask.sum()} gaussians have opacity < {opacity_thre}; "
            f"setting labels to IGNORE_INDEX={IGNORE_INDEX}."
        )
    invalid_mask = dist_exceed_mask | low_opacity_mask
    normal_gs = mesh_normals[nn_idx, :]

    if seg_indices is not None:
        segIndex_gs = seg_indices[nn_idx]
    else:
        segIndex_gs = np.zeros_like(nn_idx)

    semantic20_gs = np.full(N_gs, IGNORE_INDEX, dtype=np.int16)
    semantic200_gs = np.full(N_gs, IGNORE_INDEX, dtype=np.int16)
    instance_gs = np.full(N_gs, IGNORE_INDEX, dtype=np.int16)

    if split_name != "test":
        for group in seg_groups:
            group_segments, label_id20, label_id200 = point_indices_from_group(
                seg_indices, group, labels_pd
            )
            mask = np.isin(segIndex_gs, group_segments)
            semantic20_gs[mask] = label_id20
            semantic200_gs[mask] = label_id200
            instance_gs[mask] = group["id"]

    if invalid_mask.any():
        semantic20_gs[invalid_mask] = IGNORE_INDEX
        semantic200_gs[invalid_mask] = IGNORE_INDEX
        instance_gs[invalid_mask] = IGNORE_INDEX
    # Optional: prune gaussians outside the mesh bbox
    if bbox_pruning:
        within_mask = oriented_bbox.get_point_indices_within_bounding_box(
            gs_o3d.points
        )
        print(
            "Pruned", len(coord_gs) - len(within_mask), "gaussians by init bounding box."
        )
    else:
        within_mask = np.arange(len(coord_gs))

    # Save language features compactly if available
    if not skip_feat and gs_feat_np is not None:
        valid_indices, lang_feat = save_compact_feature(
            gs_feat_np,
            gs_feat_index,
            within_mask,
            output_path,
            "lang_feat",
            write_valid_mask=True,
        )
        print(f"[{scene_id}] saved lang_feat with {len(valid_indices)}/{len(within_mask)} valid rows.")

    # DINOv3 features (optional)
    if dino_root is not None:
        dino_path, dino_index_path = resolve_ludvig_feature_paths(
            dino_root, scene_id, "dinov3", "dino_feat"
        )
        if dino_path is not None and dino_path.exists():
            dino_feat, dino_index = load_npy_feature(dino_path, dino_index_path)
            dino_indices, dino_feat = save_compact_feature(
                dino_feat, dino_index, within_mask, output_path, "dino_feat"
            )
            print(f"[{scene_id}] saved dino_feat with {len(dino_indices)}/{len(within_mask)} valid rows.")

    # PE-Spatial features (optional)
    if pe_root is not None:
        pe_path, pe_index_path = resolve_ludvig_feature_paths(
            pe_root, scene_id, "pe_spatial", "pe_feat"
        )
        if pe_path is not None and pe_path.exists():
            pe_feat, pe_index = load_npy_feature(pe_path, pe_index_path)
            pe_indices, _ = save_compact_feature(
                pe_feat, pe_index, within_mask, output_path, "pe_feat"
            )
            print(f"[{scene_id}] saved pe_feat with {len(pe_indices)}/{len(within_mask)} valid rows.")

    # Save outputs
    np.save(os.path.join(output_path, "coord.npy"), coord_gs[within_mask].astype(np.float16))
    np.save(os.path.join(output_path, "color.npy"), color_gs[within_mask].astype(np.uint8))
    np.save(os.path.join(output_path, "opacity.npy"), opacity_gs[within_mask].astype(np.float16))
    np.save(os.path.join(output_path, "scale.npy"), scale_gs[within_mask].astype(np.float16))
    np.save(os.path.join(output_path, "quat.npy"), quat_gs[within_mask].astype(np.float16))
    np.save(os.path.join(output_path, "normal.npy"), normal_gs[within_mask].astype(np.float16))
    if split_name != "test":
        np.save(os.path.join(output_path, "segment20.npy"), semantic20_gs[within_mask].astype(np.int32))
        np.save(
            os.path.join(output_path, "segment200.npy"), semantic200_gs[within_mask].astype(np.int32)
        )
        np.save(os.path.join(output_path, "instance.npy"), instance_gs[within_mask].astype(np.int32))

    # Save PC-level data only for validation split
    if split_name == "val" and seg_indices is not None:
        # Compute per-vertex labels and instance ids
        pc_instance = np.full(mesh_coords.shape[0], IGNORE_INDEX, dtype=np.int32)
        pc_segment20 = np.full(mesh_coords.shape[0], IGNORE_INDEX, dtype=np.int32)
        pc_segment200 = np.full(mesh_coords.shape[0], IGNORE_INDEX, dtype=np.int32)
        for group in seg_groups:
            group_segments, label_id20, label_id200 = point_indices_from_group(
                seg_indices, group, labels_pd
            )
            mask_pc = np.isin(seg_indices, group_segments)
            pc_segment20[mask_pc] = label_id20
            pc_segment200[mask_pc] = label_id200
            pc_instance[mask_pc] = group["id"]

        np.save(os.path.join(output_path, "pc_coord.npy"), mesh_coords.astype(np.float16))
        np.save(os.path.join(output_path, "pc_instance.npy"), pc_instance.astype(np.int32))
        np.save(os.path.join(output_path, "pc_segment20.npy"), pc_segment20.astype(np.int32))
        np.save(os.path.join(output_path, "pc_segment200.npy"), pc_segment200.astype(np.int32))

    print(f"Scene {scene_id} processed successfully!")


###############################################################################
# 3) Main
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the standard ScanNet dataset containing scene folders (ignored for test split)",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val/test folders will be located",
    )
    parser.add_argument(
        "--gs_root",
        required=True,
        help="Released base NPY 3DGS root, or older raw Gaussian PLY root.",
    )
    parser.add_argument(
        "--feat_root",
        help="Path to language features.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Number of workers for preprocessing.",
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
        help="Enlargement margin added to each side when bbox pruning is enabled.",
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
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Process a single split; defaults to processing all splits.",
    )
    config = parser.parse_args()

    # For train/val, load scene paths from the dataset_root based on meta files.
    script_dir = Path(os.path.dirname(__file__))
    meta_root = script_dir / "meta_data"

    # Load label map
    labels_pd = pd.read_csv(
        meta_root / "scannetv2-labels.combined.tsv",
        sep="\t",
        header=0,
    )

    # Load train/val splits
    with open(meta_root / "scannetv2_train.txt") as train_file:
        train_scenes = train_file.read().splitlines()
    with open(meta_root / "scannetv2_val.txt") as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    if config.split == "test":
        patterns = ["/scans_test/scene*"]
        scene_paths = sorted(
            path for pattern in patterns for path in glob.glob(config.dataset_root + pattern)
        )
    else:
        patterns = ["/scans/scene*"]
        scene_paths = sorted(
            path for pattern in patterns for path in glob.glob(config.dataset_root + pattern)
        )
        if config.split == "train":
            train_set = set(train_scenes)
            scene_paths = [p for p in scene_paths if os.path.basename(p) in train_set]
        elif config.split == "val":
            val_set = set(val_scenes)
            scene_paths = [p for p in scene_paths if os.path.basename(p) in val_set]
        elif config.split is None:
            test_paths = sorted(
                path
                for pattern in ["/scans_test/scene*"]
                for path in glob.glob(config.dataset_root + pattern)
            )
            scene_paths = scene_paths + test_paths
    scene_paths = sorted(scene_paths)
    print(f"Found in total {len(scene_paths)} scenes under {config.dataset_root}")

    print("Processing scenes...")
    parallel = True
    if parallel:
        pool = ProcessPoolExecutor(max_workers=config.num_workers)
        try:
            list(
                pool.map(
                    handle_process,
                    scene_paths,
                    repeat(config.output_root),
                    repeat(labels_pd),
                    repeat(train_scenes),
                    repeat(val_scenes),
                    repeat(config.gs_root),
                    repeat(config.feat_root),
                    repeat(config.feat_only),
                    repeat(config.skip_feat),
                    repeat(config.bbox_pruning),
                    repeat(config.bbox_enlargement),
                    repeat(config.dino_root),
                    repeat(config.pe_root),
                    repeat(config.nn_dist_thre),
                    repeat(config.opacity_thre),
                )
            )
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, cleaning up...", flush=True)
            cancel_and_terminate_pool(pool)
            raise SystemExit(130)
        else:
            pool.shutdown()
    else:
        for scene_path in scene_paths:
            handle_process(
                scene_path,
                config.output_root,
                labels_pd,
                train_scenes,
                val_scenes,
                config.gs_root,
                config.feat_root,
                config.feat_only,
                config.skip_feat,
                config.bbox_pruning,
                config.bbox_enlargement,
                config.dino_root,
                config.pe_root,
                config.nn_dist_thre,
                config.opacity_thre,
            )
    print("Finish processing all ScanNet scenes!")
