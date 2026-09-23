import os
import argparse
import glob
import shutil
import numpy as np
import torch
import json
import csv
import cv2
import struct
import collections
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from pointcept.utils.misc import _majority_vote

from PIL import Image as PILImage
from scripts.encode_labels import SigLIPLabelEncoder
from tools.gaussian_renderer import render_semantic_mask
from tools.overall_metrics import write_overall_metrics
from tools.lerf_coloured_masks import save_coloured_mask_results

# Canonical reference concepts

CANONICAL_PROMPTS = [
    "object",
    "things",
    "stuff",
    "texture",
]

DEFAULT_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.80)


# COLMAP Parser reuse from Gaussian Splatting repo

# COLMAP Camera Structures

CameraModel = collections.namedtuple(
    "CameraModel",
    ["model_id", "model_name", "num_params"]
)

Camera = collections.namedtuple(
    "Camera",
    ["id", "model", "width", "height", "params"]
)

BaseImage = collections.namedtuple(
    "Image",
    [
        "id",
        "qvec",
        "tvec",
        "camera_id",
        "name",
        "xys",
        "point3D_ids",
    ],
)

CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_IDS = {
    model.model_id: model
    for model in CAMERA_MODELS
}
# Quaternion → Rotation Matrix

def qvec2rotmat(qvec):

    return np.array([
        [
            1 - 2*qvec[2]**2 - 2*qvec[3]**2,
            2*qvec[1]*qvec[2] - 2*qvec[0]*qvec[3],
            2*qvec[3]*qvec[1] + 2*qvec[0]*qvec[2],
        ],
        [
            2*qvec[1]*qvec[2] + 2*qvec[0]*qvec[3],
            1 - 2*qvec[1]**2 - 2*qvec[3]**2,
            2*qvec[2]*qvec[3] - 2*qvec[0]*qvec[1],
        ],
        [
            2*qvec[3]*qvec[1] - 2*qvec[0]*qvec[2],
            2*qvec[2]*qvec[3] + 2*qvec[0]*qvec[1],
            1 - 2*qvec[1]**2 - 2*qvec[2]**2,
        ],
    ])


class Image(BaseImage):

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


# Read Binary Bytes

def read_next_bytes(
    fid,
    num_bytes,
    format_char_sequence,
    endian_character="<",
):

    data = fid.read(num_bytes)

    return struct.unpack(
        endian_character + format_char_sequence,
        data,
    )

# Read COLMAP Cameras

def read_intrinsics_binary(path_to_model_file):

    cameras = {}

    with open(path_to_model_file, "rb") as fid:

        num_cameras = read_next_bytes(
            fid,
            8,
            "Q",
        )[0]

        for _ in range(num_cameras):

            camera_properties = read_next_bytes(
                fid,
                24,
                "iiQQ",
            )

            camera_id = camera_properties[0]

            model_id = camera_properties[1]

            model_name = CAMERA_MODEL_IDS[
                model_id
            ].model_name

            width = camera_properties[2]
            height = camera_properties[3]

            num_params = CAMERA_MODEL_IDS[
                model_id
            ].num_params

            params = read_next_bytes(
                fid,
                8 * num_params,
                "d" * num_params,
            )

            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )

    return cameras

# Read COLMAP Images

def read_extrinsics_binary(path_to_model_file):

    images = {}

    with open(path_to_model_file, "rb") as fid:

        num_images = read_next_bytes(
            fid,
            8,
            "Q",
        )[0]

        for _ in range(num_images):

            binary = read_next_bytes(
                fid,
                64,
                "idddddddi",
            )

            image_id = binary[0]

            qvec = np.array(binary[1:5])

            tvec = np.array(binary[5:8])

            camera_id = binary[8]

            image_name = ""

            current_char = read_next_bytes(
                fid,
                1,
                "c",
            )[0]

            while current_char != b"\x00":

                image_name += current_char.decode("utf-8")

                current_char = read_next_bytes(
                    fid,
                    1,
                    "c",
                )[0]

            num_points2D = read_next_bytes(
                fid,
                8,
                "Q",
            )[0]

            x_y_id_s = read_next_bytes(
                fid,
                24 * num_points2D,
                "ddq" * num_points2D,
            )

            xys = np.column_stack([
                tuple(map(float, x_y_id_s[0::3])),
                tuple(map(float, x_y_id_s[1::3])),
            ])

            point3D_ids = np.array(
                tuple(map(int, x_y_id_s[2::3]))
            )

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )

    return images


# Load COLMAP Camera Data

def load_camera_data(colmap_sparse_folder):

    print("\n[INFO] Loading COLMAP camera data...\n")

    cameras = read_intrinsics_binary(
        os.path.join(
            colmap_sparse_folder,
            "cameras.bin",
        )
    )

    images = read_extrinsics_binary(
        os.path.join(
            colmap_sparse_folder,
            "images.bin",
        )
    )

    print(f"Loaded cameras : {len(cameras)}")
    print(f"Loaded images  : {len(images)}\n")

    return cameras, images


# Argument Parser

def parse_args():

    parser = argparse.ArgumentParser(
        description="SceneSplat Evaluation on LERF"
    )
    

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="SceneSplat checkpoint (used in later milestones)"
    )

    parser.add_argument(
        "--input_root",
        required=True,
        help="Folder containing preprocessed SceneSplat .npy files"
    )

    parser.add_argument(
        "--feature_path",
        required=True,
        help="SceneSplat feature tensor (.pt)"
    )

    parser.add_argument(
        "--label_root",
        required=True,
        help="LERF annotation folder"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output root; writes lerf_new_runs/results_XX/<scene>"
    )

    parser.add_argument(
        "--image_root",
        required=True,
        help="LERF RGB image folder"
    )

    parser.add_argument(
        "--colmap_sparse",
        required=True,
        help="COLMAP sparse/0 folder"
    )

    parser.add_argument(
        "--query",
        default=None,
        help="Evaluate only one category for debugging; default evaluates all."
    )


    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
        help="Fixed relevance thresholds for every query (default: 0.20 0.30 0.40 0.50 0.60 0.80)"
    )


    args = parser.parse_args()
    if not args.thresholds or any(not np.isfinite(t) or not 0 <= t <= 1 for t in args.thresholds):
        parser.error("thresholds must be finite and in [0, 1]")
    if len(set(args.thresholds)) != len(args.thresholds):
        parser.error("thresholds must be unique")
    if any(not np.isclose(round(t * 100), t * 100) for t in args.thresholds):
        parser.error("thresholds must be multiples of 0.01 for results_XX folder names")
    return args

# Helper Functions

def load_camera_image(image_path):

    image = np.array(PILImage.open(image_path))

    return image


def build_ground_truth_masks(annotation, image):

    H, W = image.shape[:2]

    gt_masks = {}

    for obj in annotation:

        category = obj["category"]
        polygon = obj["polygon"]

        if category not in gt_masks:
            gt_masks[category] = np.zeros((H, W), dtype=np.uint8)

        gt_masks[category] |= polygon_to_mask(
            polygon,
            H,
            W
        )

    return gt_masks


# Load Preprocessed Scene

def load_preprocessed_scene(input_root):

    print("\n[INFO] Loading preprocessed scene...\n")

    scene_data = {}
    
    for file in sorted(os.listdir(input_root)):

        if file.endswith(".npy"):

            key = os.path.splitext(file)[0]

            scene_data[key] = np.load(
                os.path.join(input_root, file)
            )

            print(
                f"Loaded {key:<20} {scene_data[key].shape}"
            )

    print()

    return scene_data


# Load SceneSplat Features

def load_scenesplat_features(feature_path):

    print("[INFO] Loading SceneSplat features...")
    print(f"Feature file:\n{feature_path}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = torch.load(
        feature_path,
        map_location=device
    )

    if isinstance(features, torch.Tensor):

        print("[INFO] Loaded Tensor.")
        print(f"Feature device      : {features.device}")

    elif isinstance(features, dict):

        print("[INFO] Loaded Dictionary.")
        print(f"Dictionary keys     : {list(features.keys())}")

    else:

        raise RuntimeError(
            f"Unsupported feature type: {type(features)}"
        )

    print()

    return features
# Load LERF Annotations
def load_lerf_annotations(annotation_dir):
    """
    Reads all LERF annotation jsons.

    Returns
    -------
    {
        image_name : [
            {
                "category": "...",
                "bbox": [...],
                "polygon": [...]
            },
            ...
        ]
    }
    """

    annotations = {}

    json_files = sorted(
        glob.glob(os.path.join(annotation_dir, "*.json"))
    )

    for jf in json_files:

        with open(jf, "r") as f:
            data = json.load(f)

        image_name = data["info"]["name"]

        objs = []

        for obj in data["objects"]:

            objs.append({
                "category": obj["category"],
                "bbox": obj["bbox"],
                "polygon": obj["segmentation"]
            })

        annotations[image_name] = objs

    return annotations


def polygon_to_mask(polygon, height, width):
    """
    Converts LabelMe polygon into binary mask.
    """

    mask = np.zeros((height, width), dtype=np.uint8)

    pts = np.array(polygon, dtype=np.int32)

    cv2.fillPoly(mask, [pts], 1)

    return mask


# Collect LERF Categories

def get_categories(annotations):
    """
    Collect all unique LERF object categories.
    """

    categories = sorted({
        obj["category"]
        for image in annotations.values()
        for obj in image
    })

    print(f"\nFound {len(categories)} unique categories:\n")

    for category in categories:
        print(f"  - {category}")

    print()

    return categories


# Build THGS-style Query Text Embeddings

def build_query_text_embeddings(
    query,
    encoder,
    canonical_prompts=CANONICAL_PROMPTS
):
    """
    Encode one query together with generic canonical concepts.

    Example:
        query = "yellow pouf"

    Encodes:
        [
            "yellow pouf",
            "object",
            "things",
            "stuff",
            "texture"
        ]

    Returns
    -------
    text_embeddings : Tensor [5, 768]

        index 0   -> query
        indices 1: -> canonical reference concepts
    """

    labels = [
        query,
        *canonical_prompts
    ]

    print("\n[INFO] Building text embeddings...")
    print(f"Query               : {query}")
    print(f"Canonical references: {canonical_prompts}")

    text_embeddings = encoder.encode_labels(
        labels
    )

    text_embeddings = torch.nn.functional.normalize(
        text_embeddings.float(),
        dim=1
    )

    print(
        f"Text embedding shape: "
        f"{tuple(text_embeddings.shape)}"
    )

    return text_embeddings


# Normalize Scene Features

def normalize_scene_features(features):

    print("\n[INFO] Normalizing SceneSplat features...\n")

    if isinstance(features, dict):

        raise RuntimeError(
            "Expected tensor features."
        )

    features = features.float()

    features = torch.nn.functional.normalize(
        features,
        dim=1
    )

    print(
        f"Normalized feature shape : {tuple(features.shape)}"
    )

    return features


# THGS-style Query Similarity

def compute_thgs_style_similarity(
    scene_features,
    text_embeddings,
    temperature=10.0
):
    """
    Compute THGS-style query similarity.

    
        Normalized embeddings for:
        [query, object, things, stuff, texture]

    """

    print(
        "\n[INFO] Computing THGS-style "
        "query similarity...\n"
    )

    # Ensure float tensors

    scene_features = scene_features.float()
    text_embeddings = text_embeddings.float()

    # Ensure both tensors are on the same device

    if scene_features.device != text_embeddings.device:
        text_embeddings = text_embeddings.to(
            scene_features.device
        )

    # Compute [N, 5] cosine scores for the query, object, things, stuff, and texture.

    logits = (
        scene_features
        @ text_embeddings.T
    )

    print(
        f"Gaussian features       : "
        f"{tuple(scene_features.shape)}"
    )

    print(
        f"Text embeddings         : "
        f"{tuple(text_embeddings.shape)}"
    )

    print(
        f"Raw similarity matrix   : "
        f"{tuple(logits.shape)}"
    )

    # Keep query scores as an [N, 1] column.

    positive_vals = logits[:, 0:1]

    # Take the [N, 4] scores for object, things, stuff, and texture.

    negative_vals = logits[:, 1:]

    num_canonical = negative_vals.shape[1]

    # Repeat each query score across the four reference concepts.

    repeated_pos = positive_vals.repeat(
        1,
        num_canonical
    )

    # Pair each query score with each reference score to form an [N, 4, 2] tensor.

    pairwise_logits = torch.stack(
        (
            repeated_pos,
            negative_vals
        ),
        dim=-1
    )

    # Apply temperature-scaled softmax; index 0 is the query and index 1 is the reference.

    pairwise_prob = torch.softmax(
        temperature * pairwise_logits,
        dim=-1
    )

    # Get the query probability against each of the four references.

    positive_prob = pairwise_prob[..., 0]

    # Use the lowest query probability across the four references as each Gaussian score.

    query_similarity = positive_prob.min(
        dim=1
    ).values

    # Debug / verification output

    print(
        f"Pairwise probabilities  : "
        f"{tuple(pairwise_prob.shape)}"
    )

    print(
        f"Final query similarity  : "
        f"{tuple(query_similarity.shape)}"
    )

    print(
        f"Final score range       : "
        f"{query_similarity.min().item():.4f} "
        f"to "
        f"{query_similarity.max().item():.4f}"
    )

    print(
        f"Final score mean        : "
        f"{query_similarity.mean().item():.4f}"
    )

    print(
        f"Temperature             : "
        f"{temperature}"
    )

    return query_similarity


# Helper for Gaussian Renderer

def get_colmap_image_and_camera(images, cameras, image_name):
    colmap_image = None

    for img in images.values():
        if img.name == image_name:
            colmap_image = img
            break

    if colmap_image is None:
        raise RuntimeError(
            f"{image_name} not found in COLMAP images."
        )

    camera = cameras[colmap_image.camera_id]

    return colmap_image, camera


# Show Top Retrieval Results

def show_top_retrievals(
    query,
    query_similarity,
    top_k=10
):

    values, indices = torch.topk(
        query_similarity,
        top_k
    )

    print("\n========== Top Retrieval ==========\n")

    print(f"Query : {query}\n")

    for rank in range(top_k):

        print(
            f"{rank+1:>2}. "
            f"Gaussian {indices[rank].item():<10}"
            f"Score {values[rank].item():.4f}"
        )

    print("\n===================================\n")


# Save Query Similarity Histogram

# Save Query Similarity Histogram

def save_similarity_histogram(
    query,
    query_similarity,
    output_dir,
    threshold=None,
    threshold_source=None
):
    """
    Save the distribution of per-Gaussian query similarity
    scores for one query.

    If a query-specific threshold is available, draw that
    threshold on the histogram.

    No Top-K cutoff is used.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    scores = (
        query_similarity
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        scores,
        bins=100
    )

    # Draw actual query-specific operating threshold

    if threshold is not None:

        label = (
            f"Threshold = {threshold:.4f}"
        )

        if threshold_source is not None:

            label += (
                f" ({threshold_source})"
            )

        plt.axvline(
            threshold,
            linestyle="--",
            label=label
        )

        plt.legend()

    plt.xlabel(
        "Query similarity"
    )

    plt.ylabel(
        "Number of Gaussians"
    )

    plt.title(
        f"Query Similarity Distribution: {query}"
    )

    plt.tight_layout()

    safe_query = query.replace(
        " ",
        "_"
    )

    filename = os.path.join(
        output_dir,
        f"{safe_query}_similarity_histogram.png"
    )

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved similarity histogram : {filename}"
    )

    if threshold is not None:

        print(
            f"Similarity threshold       : "
            f"{threshold:.6f}"
        )

        if threshold_source is not None:

            print(
                f"Threshold source           : "
                f"{threshold_source}"
            )

    return filename


# Save Query Similarity Scores

def save_query_similarity(
    query,
    query_similarity,
    output_dir
):
    """
    Save one continuous query similarity score
    for every Gaussian.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    scores = (
        query_similarity
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    safe_query = query.replace(" ", "_")

    filename = os.path.join(
        output_dir,
        f"{safe_query}_similarity.npy"
    )
    np.save(
        filename,
        scores
    )

    print(
        f"Saved query similarities : {filename}"
    )

    return filename

# Build Continuous Similarity Colors

def build_similarity_colors(
    query_similarity
):
    """
    Convert continuous query similarities to RGB colors.

    Every Gaussian is retained.
    No thresholding or Top-K selection is performed here.
    """

    scores = (
        query_similarity
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    score_min = float(
        scores.min()
    )

    score_max = float(
        scores.max()
    )

    # Min-max normalization only for color visualization.
    if score_max > score_min:

        normalized = (
            scores - score_min
        ) / (
            score_max - score_min
        )

    else:

        normalized = np.zeros_like(
            scores
        )

    cmap = plt.get_cmap(
        "viridis"
    )

    rgba = cmap(
        normalized
    )

    colors = (
        rgba[:, :3] * 255
    ).astype(
        np.uint8
    )

    return colors

# Save Similarity-Colored Point Cloud

def save_similarity_ply(
    query,
    coord,
    query_similarity,
    output_dir
):
    """
    Save all Gaussians as a similarity-colored PLY point cloud.

    XYZ comes from SceneSplat Gaussian coordinates.
    RGB represents continuous query similarity.

    No thresholding is performed.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    xyz = np.asarray(
        coord,
        dtype=np.float32
    )

    colors = build_similarity_colors(
        query_similarity
    )

    if xyz.shape[0] != colors.shape[0]:

        raise RuntimeError(
            f"Coordinate count ({xyz.shape[0]}) "
            f"does not match similarity count "
            f"({colors.shape[0]})."
        )

    safe_query = query.replace(
        " ",
        "_"
    )

    filename = os.path.join(
        output_dir,
        f"{safe_query}_similarity.ply"
    )

    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )

    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]

    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(
        filename,
        "wb"
    ) as f:

        f.write(
            header.encode("ascii")
        )

        vertex.tofile(
            f
        )

    print(
        f"Saved similarity-colored PLY : {filename}"
    )

    return filename


# Similarity Threshold Gaussian Selection

def threshold_gaussian_selection(
    query_similarity,
    threshold
):
    """
    Select all Gaussians whose query similarity is
    greater than or equal to the given threshold.
    """

    selected_mask = (
        query_similarity >= threshold
    )

    selected_indices = torch.nonzero(
        selected_mask,
        as_tuple=False
    ).squeeze(1)

    selected_scores = query_similarity[
        selected_indices
    ]

    if selected_scores.numel() > 0:

        order = torch.argsort(
            selected_scores,
            descending=True
        )

        selected_indices = selected_indices[
            order
        ]

        selected_scores = selected_scores[
            order
        ]

    return (
        selected_indices,
        selected_scores
    )

# Prepare Evaluation Frames For One Query

def prepare_query_evaluation_frames(
    query,
    annotations,
    image_root,
    colmap_images,
    cameras
):
    """
    Load all annotated frames containing the query once.

    These cached frames are reused during threshold search
    and final evaluation.
    """

    frames = []

    for image_name, annotation in annotations.items():

        image_path = os.path.join(
            image_root,
            image_name
        )

        image = load_camera_image(
            image_path
        )

        gt_masks = build_ground_truth_masks(
            annotation,
            image
        )

        if query not in gt_masks:
            continue

        gt_mask = gt_masks[
            query
        ]

        colmap_image, camera = (
            get_colmap_image_and_camera(
                colmap_images,
                cameras,
                image_name
            )
        )

        frames.append({
            "image_name": image_name,
            "image": image,
            "gt_mask": gt_mask,
            "colmap_image": colmap_image,
            "camera": camera,
        })

    print(
        f"[THRESHOLD] Query '{query}' has "
        f"{len(frames)} GT evaluation frames."
    )

    return frames


def render_query_mask(scene_data, selected_indices, colmap_image, camera, image_shape):
    """Render the same selected Gaussians for annotated and unannotated frames."""
    if len(selected_indices) == 0:
        return (np.zeros(image_shape, dtype=np.uint8),
                np.zeros(image_shape, dtype=np.float32))

    return render_semantic_mask(
        coord=scene_data["coord"],
        scale=scene_data["scale"],
        quat=scene_data["quat"],
        opacity=scene_data["opacity"],
        selected_indices=selected_indices,
        image=colmap_image,
        camera=camera,
        image_shape=image_shape,
        mask_threshold=0.05,
        sigma_extent=3.0,
        max_radius=60,
        opacity_threshold=0.01,
    )


# Find Best Threshold For One Query

def build_gaussian_knn_graph(
    coord,
    vote_k=25
):
    """
    Build the SceneSplat-style spatial KNN graph once.

    For every Gaussian center, find its vote_k nearest
    Gaussian centers in 3D.

    Returns
    -------
    knn_indices : np.ndarray [N, vote_k]
        Neighbor indices for every Gaussian.

    Notes
    -----
    cKDTree.query includes the query point itself as the
    nearest neighbor when querying the same coordinate set.
    This matches the behavior of SceneSplat's native voting.
    """

    print(
        "\n========== Building Gaussian KNN Graph =========="
    )

    coord_np = np.asarray(
        coord,
        dtype=np.float32
    )

    print(
        f"Gaussian count : {coord_np.shape[0]}"
    )

    print(
        f"Vote K         : {vote_k}"
    )

    kd_tree = cKDTree(
        coord_np
    )

    _, knn_indices = kd_tree.query(
        coord_np,
        k=vote_k,
        workers=-1
    )

    if vote_k == 1:

        knn_indices = knn_indices[
            :,
            None
        ]

    knn_indices = np.asarray(
        knn_indices,
        dtype=np.int64
    )

    print(
        f"KNN graph shape: {knn_indices.shape}"
    )

    print(
        "=================================================\n"
    )

    return knn_indices


# SceneSplat-style Binary Neighbor Voting

def apply_gaussian_neighbor_voting(
    selected_indices,
    num_gaussians,
    knn_indices
):
    """
    Apply SceneSplat-style majority voting to the current
    binary query prediction.

    Labels:
        0 -> background
        1 -> query-positive

    Parameters
    ----------
    selected_indices
        Gaussian indices selected by the similarity threshold.

    num_gaussians
        Total number of Gaussians.

    knn_indices
        Precomputed [N, K] nearest-neighbor graph.

    Returns
    -------
    voted_indices : np.ndarray
        Gaussian indices classified as query-positive after
        KNN majority voting.
    """

    if torch.is_tensor(
        selected_indices
    ):

        selected_indices_np = (
            selected_indices
            .detach()
            .cpu()
            .numpy()
        )

    else:

        selected_indices_np = np.asarray(
            selected_indices,
            dtype=np.int64
        )

    # Convert the threshold mask to labels: 0 for background and 1 for the query.

    initial_labels = np.zeros(
        num_gaussians,
        dtype=np.int32
    )

    initial_labels[
        selected_indices_np
    ] = 1

    # Vote over [N, K] neighbor labels using two classes; no points have the ignore label.

    neighbor_labels = initial_labels[
        knn_indices
    ]

    voted_labels = _majority_vote(
        neighbor_labels=neighbor_labels,
        ignore_label=-1,
        num_classes=2
    )

    voted_indices = np.flatnonzero(
        voted_labels == 1
    ).astype(
        np.int64
    )

    return voted_indices


# Preview Selected Gaussians

def preview_selected_gaussians(
    indices,
    scores,
    num_examples=10
):

    print("\n========== Selected Gaussians ==========\n")

    num_examples = min(
        num_examples,
        len(indices)
    )

    for i in range(num_examples):

        print(
            f"Gaussian {indices[i].item():<8}"
            f"Score {scores[i].item():.4f}"
        )

    print("\n========================================\n")

# Save Retrieval Results

def save_retrieval_results(
    output_dir,
    query,
    selected_indices,
    selected_scores,
    threshold,
    total_gaussians,
    threshold_source
):
    """
    Save retrieval results for one query as JSON.
    """

    print("\n[INFO] Saving retrieval results...\n")

    os.makedirs(output_dir, exist_ok=True)
    selected_fraction = len(selected_indices) / total_gaussians
    result = {

        "query": query,

        "selection_method": (
            "fixed_query_similarity_threshold_"
            "with_knn_majority_voting"
        ),

        "neighbor_voting": True,

        "vote_k": 25,

        "threshold": float(
            threshold
        ),

        "threshold_source": (
            threshold_source
        ),

        "num_selected": int(
            len(selected_indices)
        ),

        "selected_fraction": float(
            selected_fraction
        ),

        "gaussian_indices": (
            selected_indices
            .detach()
            .cpu()
            .tolist()
        ),

        "similarity_scores": (
            selected_scores
            .detach()
            .cpu()
            .tolist()
        )
    }


    output_file = os.path.join(
        output_dir,
        f"{query}_retrieval.json"
    )

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved retrieval results to:")
    print(output_file)

    return output_file

# Build Visualization Mask

def build_visualization_mask(
    num_gaussians,
    selected_indices
):
    """
    Creates a binary mask indicating which Gaussians
    were selected for a query.
    """

    mask = torch.zeros(
        num_gaussians,
        dtype=torch.uint8
    )

    indices_cpu = (
        selected_indices
        .detach()
        .cpu()
        .long()
    )

    mask[indices_cpu] = 1

    return mask


# Save Visualization Mask

def save_visualization_mask(
    output_dir,
    query,
    mask
):
    """
    Save binary visualization mask for one query.
    """

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(
        output_dir,
        f"{query}_mask.npy"
    )

    np.save(filename, mask.cpu().numpy())

    print(f"Saved mask : {filename}")

    return filename
# Build Gaussian Visualization Colors

def build_visualization_colors(
    num_gaussians,
    selected_indices
):
    """
    Build RGB colors for visualization.

    Selected Gaussians:
        Red

    Remaining Gaussians:
        Light Gray
    """

    colors = np.full(
        (num_gaussians, 3),
        180,
        dtype=np.uint8
    )

    indices_cpu = (
        selected_indices
        .detach()
        .cpu()
        .numpy()
    )

    colors[indices_cpu] = np.array(
        [255, 0, 0],
        dtype=np.uint8
    )

    return colors


def evaluate_iou(
    prediction_mask,
    ground_truth_mask
):
    """
    Compute IoU.
    """

    prediction = prediction_mask.astype(bool)
    groundtruth = ground_truth_mask.astype(bool)

    intersection = np.logical_and(
        prediction,
        groundtruth
    ).sum()

    union = np.logical_or(
        prediction,
        groundtruth
    ).sum()

    if union == 0:
        return 0.0

    return intersection / union


def binary_macc(prediction, target):
    """Mean pixel accuracy of foreground and background classes present in GT."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    accuracies = [np.mean(prediction[target == label] == label)
                  for label in (False, True) if np.any(target == label)]
    return float(np.mean(accuracies))
    
# Save Visualization Colors

def save_visualization_colors(
    output_dir,
    query,
    colors
):
    """
    Save RGB visualization colors.
    """

    filename = os.path.join(
        output_dir,
        f"{query}_colors.npy"
    )

    np.save(filename, colors)

    print(f"Saved colors : {filename}")

    return filename

def save_mask_visualization(
    image,
    gt_mask,
    prediction_mask,
    output_dir,
    query,
    image_name
):
    """
    Save visualization comparing

    1. RGB image
    2. Ground-truth mask
    3. Predicted mask
    4. Overlay
    """


    folder = "mask_visualizations"

    os.makedirs(
        os.path.join(output_dir, folder),
        exist_ok=True
    )

    overlay = image.copy()

    # Ground Truth = Green
    overlay[gt_mask > 0] = (
        0.6 * overlay[gt_mask > 0]
        + 0.4 * np.array([0,255,0])
    ).astype(np.uint8)

    # Prediction = Red
    overlay[prediction_mask > 0] = (
        0.6 * overlay[prediction_mask > 0]
        + 0.4 * np.array([255,0,0])
    ).astype(np.uint8)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18,5)
    )

    axes[0].imshow(image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask,cmap="gray",vmin=0,vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction_mask,cmap="gray",vmin=0,vmax=1)
    coverage = 100.0 * np.count_nonzero(prediction_mask) / prediction_mask.size
    axes[2].set_title(f"Prediction ({coverage:.1f}% pixels)")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()

    filename = os.path.join(
        output_dir,
        folder,
        f"{query}_{image_name}.png"
    )

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved visualization : {filename}")

    return filename
# Save Evaluation Summary

def save_evaluation_summary(
    output_dir,
    queries,
    retrieval_files,
    mask_files,
    color_files
):
    """
    Save overall evaluation summary.
    """

    summary = {
        "num_queries": len(queries),
        "queries": queries,
        "retrieval_files": retrieval_files,
        "mask_files": mask_files,
        "color_files": color_files
    }

    filename = os.path.join(
        output_dir,
        "evaluation_summary.json"
    )

    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved evaluation summary : {filename}")


# Save Per Query CSV

def save_per_query_csv(
    output_dir,
    per_query_results
):

    filename = os.path.join(
        output_dir,
        "per_query_results.csv"
    )

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Method",
            "Scene",
            "Query",
            "Mean_IoU",
            "mAcc",
            "Threshold",
            "Threshold_Source",
            "Raw_Selected_Gaussians",
            "Voted_Selected_Gaussians"
        ])

        for row in per_query_results:

            writer.writerow([
                row["method"],
                row["scene"],
                row["query"],

                round(
                    row["mean_iou"],
                    6
                ),

                round(row["mAcc"], 6),

                round(
                    row["threshold"],
                    6
                ),

                row["threshold_source"],

                row["raw_selected_gaussians"],
                row["voted_selected_gaussians"]
            ])

    print(f"Saved CSV : {filename}")

    return filename

# Save Image-wise IoU Results

def save_imagewise_results(
    output_dir,
    query,
    image_results
):
    """
    Save IoU for every image of a query.

    Parameters
    ----------
    image_results : list of tuples

        [
            (image_name, IoU),
            ...
        ]
    """

    filename = os.path.join(
        output_dir,
        f"{query}_image_results.csv"
    )

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["Image", "IoU", "mAcc"])

        for image_name, iou, macc in image_results:

            writer.writerow([
                image_name,
                round(iou, 6),
                round(macc, 6)
            ])

    print(f"Saved image-wise results : {filename}")

    return filename


# Compute Global Evaluation Metrics

def compute_global_metrics(
    query_metrics,
    query_macc_metrics
):

    # Require at least one evaluated query before computing metrics.
    if len(query_metrics) == 0:
        raise RuntimeError(
            "No query metrics were computed."
        )

    values = np.array(
        list(query_metrics.values()),
        dtype=np.float32
    )

    metrics = {}

    metrics["queries"] = len(values)

    metrics["mean_iou"] = float(np.mean(values))
    metrics["mAcc"] = float(np.mean(list(query_macc_metrics.values())))

    metrics["median_iou"] = float(np.median(values))

    metrics["std_iou"] = float(np.std(values))

    metrics["min_iou"] = float(np.min(values))

    metrics["max_iou"] = float(np.max(values))

    metrics["precision@0.25"] = float(
        np.mean(values >= 0.25)
    )

    metrics["precision@0.50"] = float(
        np.mean(values >= 0.50)
    )

    return metrics

# Save Global Metrics

def save_global_metrics(
    output_dir,
    global_metrics
):

    filename = os.path.join(
        output_dir,
        "evaluation_metrics.json"
    )

    with open(filename, "w") as f:

        json.dump(
            global_metrics,
            f,
            indent=4
        )

    print(f"Saved metrics : {filename}")

    overall_filename = write_overall_metrics(os.path.dirname(output_dir))
    print(f"Saved overall metrics : {overall_filename}")

    return filename


# Inspect first LERF Annotations
def inspect_first_annotation(annotations):

    first_image = next(iter(annotations))

    print(f"Image : {first_image}\n")

    for obj in annotations[first_image]:

        print(
            f"{obj['category']:<25}"
            f"1 polygon"
        )

    print("\n=====================================\n")


# Print Evaluation Statistics

def print_evaluation_statistics(
    queries,
    num_gaussians,
    retrieval_files,
    mask_files,
    color_files
):

    print("\n========== Evaluation Statistics ==========\n")

    print(f"Queries evaluated      : {len(queries)}")
    print(f"Scene Gaussians        : {num_gaussians}")
    print(f"Retrieval files        : {len(retrieval_files)}")
    print(f"Visualization masks    : {len(mask_files)}")
    print(f"Visualization colors   : {len(color_files)}")

    print("\n===========================================\n")

# Fixed-threshold evaluation

def run_evaluation(args, threshold, output_dir, shared):

    scene_name = os.path.basename(
            os.path.normpath(args.input_root)
        )
    scene_output_dir = os.path.join(
        output_dir,
        scene_name
    )

    os.makedirs(
        scene_output_dir,
        exist_ok=True
    )

    if "scene_data" not in shared:
        shared["scene_data"] = load_preprocessed_scene(args.input_root)
    scene_data = shared["scene_data"]
    required = [
        "coord",
        "color",
        "opacity",
        "quat",
        "scale"
    ]

    missing = [
        key
        for key in required
        if key not in scene_data
    ]

    if missing:
        raise RuntimeError(
            f"Missing required scene arrays: {missing}"
        )


    # SceneSplat-style spatial voting graph

    vote_k = 25

    if "knn_indices" not in scene_data:
        scene_data["knn_indices"] = build_gaussian_knn_graph(
            coord=scene_data["coord"], vote_k=vote_k
        )

    print(
        f"[VOTING] SceneSplat-style neighbor voting "
        f"enabled with k={vote_k}"
    )


    if "features" not in shared:
        shared["features"] = normalize_scene_features(
            load_scenesplat_features(args.feature_path)
        )
    features = shared["features"]
    query_metrics = {}
    query_macc_metrics = {}
    all_frame_ious = []
    all_frame_maccs = []
    coloured_entries = []
    per_query_results = []


    global_metrics = {

        "scene": scene_name,

        "method": "SceneSplat",

        "similarity_method": (
            "THGS-style query-vs-canonical"
        ),

        "canonical_prompts": (
            CANONICAL_PROMPTS
        ),

        "selection_mode": (
            "fixed_similarity_threshold_with_knn_voting"
        ),

        "neighbor_voting": True,
        "vote_k": vote_k,

        "threshold": float(threshold),
        "threshold_source": "fixed",

        "per_query": {},
        "per_query_mAcc": {},
        "per_query_metrics": {},
        "mAcc_definition": "Mean of foreground and background pixel accuracy per annotated query frame; absent GT classes are excluded; macro average over frames, then queries."
    }


    if "annotations" not in shared:
        shared["annotations"] = load_lerf_annotations(args.label_root)
    annotations = shared["annotations"]

    if "cameras" not in shared:
        shared["cameras"], shared["images"] = load_camera_data(args.colmap_sparse)
    cameras, images = shared["cameras"], shared["images"]


    inspect_first_annotation(annotations)

    categories = get_categories(
        annotations
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n[INFO] Loading SceneSplat "
        "SigLIP text encoder...\n"
    )

    if "text_encoder" not in shared:
        shared["text_encoder"] = SigLIPLabelEncoder(device=device)
    text_encoder = shared["text_encoder"]


    print("\n========== Open Vocabulary Retrieval ==========\n")

    retrieval_files = []
    mask_files = []
    color_files = []


    if args.query is not None:
        if args.query not in categories:
            raise RuntimeError(
                f"Query '{args.query}' not found. "
                f"Available queries: {categories}"
            )
        queries_to_evaluate = [args.query]
    else:
        queries_to_evaluate = categories

    for query in queries_to_evaluate:

        print("\n==========================================")
        print(f"Processing query : {query}")
        print("==========================================\n")


        query_output_dir = os.path.join(
            output_dir,
            scene_name,
            query.replace(" ", "_")
        )

        os.makedirs(
            query_output_dir,
            exist_ok=True
        )
        # Encode the query followed by object, things, stuff, and texture.

        if query not in shared.setdefault("query_similarities", {}):
            query_text_embeddings = build_query_text_embeddings(
                query=query, encoder=text_encoder, canonical_prompts=CANONICAL_PROMPTS
            )
            shared["query_similarities"][query] = compute_thgs_style_similarity(
                scene_features=features, text_embeddings=query_text_embeddings,
                temperature=10.0
            )
        query_similarity = shared["query_similarities"][query]

        # Visualize similarity scores before selecting Gaussians.

        save_query_similarity(
            query=query,
            query_similarity=query_similarity,
            output_dir=query_output_dir
        )


        save_similarity_ply(
            query=query,
            coord=scene_data["coord"],
            query_similarity=query_similarity,
            output_dir=query_output_dir
        )


        show_top_retrievals(
            query,
            query_similarity
        )

        
        # Prepare annotated frames for metrics and mask visualization

        evaluation_frames = (
            prepare_query_evaluation_frames(
                query=query,
                annotations=annotations,
                image_root=args.image_root,
                colmap_images=images,
                cameras=cameras
            )
        )


        # Apply the same preset threshold to every query.
        threshold_source = "fixed"

        # Save the similarity histogram with the threshold used for this query.

        save_similarity_histogram(
            query=query,
            query_similarity=query_similarity,
            output_dir=query_output_dir,
            threshold=threshold,
            threshold_source=threshold_source
        )


        # Final Gaussian selection using fixed threshold

        selected_indices, selected_scores = (
            threshold_gaussian_selection(
                query_similarity=query_similarity,
                threshold=threshold
            )
        )
        # Apply SceneSplat-style k=25 spatial voting

        selected_indices_np = apply_gaussian_neighbor_voting(
            selected_indices=selected_indices,
            num_gaussians=query_similarity.shape[0],
            knn_indices=scene_data["knn_indices"]
        )

        selected_indices_voted = torch.as_tensor(
            selected_indices_np,
            dtype=torch.long,
            device=query_similarity.device
        )

        selected_scores_voted = query_similarity[
            selected_indices_voted
        ]

        if selected_scores_voted.numel() > 0:

            order = torch.argsort(
                selected_scores_voted,
                descending=True
            )

            selected_indices_voted = (
                selected_indices_voted[order]
            )

            selected_scores_voted = (
                selected_scores_voted[order]
            )

        print(
            f"[VOTING] Raw selected Gaussians   : "
            f"{len(selected_indices)}"
        )

        print(
            f"[VOTING] Voted selected Gaussians : "
            f"{len(selected_indices_np)}"
        )

        # Handle queries that select no Gaussians.

        if len(selected_indices_np) == 0:

            print(
                f"[WARNING] Query '{query}' selected zero Gaussians "
                f"with threshold {threshold:.6f}."
            )

            print(
                f"[WARNING] Current score range: "
                f"{query_similarity.min().item():.6f} to "
                f"{query_similarity.max().item():.6f}"
            )

        print(
            f"\n[FINAL SELECTION] Query     : {query}"
        )

        print(
            f"[FINAL SELECTION] Threshold : "
            f"{threshold:.6f}"
        )

        print(
            f"[FINAL SELECTION] Source    : "
            f"{threshold_source}"
        )

        print(
            f"[FINAL SELECTION] Raw Gaussians   : "
            f"{len(selected_indices)}"
        )

        print(
            f"[FINAL SELECTION] Voted Gaussians : "
            f"{len(selected_indices_np)}\n"
        )
        # Evaluate selected Gaussians

        query_ious = []
        query_maccs = []
        image_results = []
        annotated_frame_names = set()
        annotated_mask_dir = os.path.join(query_output_dir, "annotated_frame_masks")
        os.makedirs(annotated_mask_dir, exist_ok=True)
        best_frame = None
        # Evaluate this query on every annotated image

        for frame_index, frame in enumerate(evaluation_frames):

            image_name = frame[
                "image_name"
            ]

            image = frame[
                "image"
            ]

            gt_mask = frame[
                "gt_mask"
            ]

            colmap_image = frame[
                "colmap_image"
            ]

            camera = frame[
                "camera"
            ]

            print(
                f"\nEvaluating image : "
                f"{image_name}"
            )

            prediction_mask, soft_prediction = render_query_mask(
                scene_data, selected_indices_np, colmap_image, camera,
                image.shape[:2]
            )

            print(
                f"[MASK] {image_name}: "
                f"binary_pixels="
                f"{int(prediction_mask.sum())}, "
                f"soft_min="
                f"{soft_prediction.min():.4f}, "
                f"soft_max="
                f"{soft_prediction.max():.4f}"
            )

            mask_dir = os.path.join(query_output_dir, "predicted_masks")
            os.makedirs(mask_dir, exist_ok=True)
            PILImage.fromarray(prediction_mask.astype(np.uint8) * 255).save(
                os.path.join(mask_dir, f"{frame_index:03d}_" + os.path.splitext(os.path.basename(image_name))[0] + ".png")
            )

            annotated_frame_names.add(image_name)
            annotated_path = os.path.join(
                annotated_mask_dir, os.path.splitext(os.path.basename(image_name))[0] + ".png")
            PILImage.fromarray(prediction_mask.astype(np.uint8) * 255).save(annotated_path)
            coloured_entries.append((query, image_name, annotated_path))

            visualization = save_mask_visualization(
                image=image,
                gt_mask=gt_mask,
                prediction_mask=prediction_mask,
                output_dir=query_output_dir,
                query=query,
                image_name=os.path.splitext(
                    image_name
                )[0],
            )

            iou = evaluate_iou(
                prediction_mask,
                gt_mask
            )
            macc = binary_macc(prediction_mask, gt_mask)

            if best_frame is None or iou > best_frame[0]:
                best_frame = (float(iou), image_name, visualization)

            query_ious.append(
                iou
            )
            query_maccs.append(macc)
            all_frame_ious.append(float(iou))
            all_frame_maccs.append(float(macc))

            image_results.append(
                (
                    image_name,
                    float(iou),
                    float(macc)
                )
            )

            print(
                f"{image_name:<25} "
                f"IoU : {iou:.4f}"
            )

        for image_name in sorted(annotations):
            if image_name in annotated_frame_names:
                continue

            image_path = os.path.join(args.image_root, image_name)
            with PILImage.open(image_path) as rgb_image:
                image_shape = (rgb_image.height, rgb_image.width)

            colmap_image, camera = get_colmap_image_and_camera(
                images, cameras, image_name
            )
            prediction_mask, _ = render_query_mask(
                scene_data, selected_indices_np, colmap_image,
                camera, image_shape
            )
            annotated_path = os.path.join(
                annotated_mask_dir, os.path.splitext(os.path.basename(image_name))[0] + ".png")
            PILImage.fromarray(prediction_mask.astype(np.uint8) * 255).save(annotated_path)
            coloured_entries.append((query, image_name, annotated_path))

        print(f"Saved {len(annotations)} annotated-frame masks to {annotated_mask_dir}")

        if best_frame is not None:
            best_dir = os.path.join(scene_output_dir, "best_visualization")
            os.makedirs(best_dir, exist_ok=True)
            best_iou, best_image, best_visualization = best_frame
            best_name = (f"{query.replace(' ', '_')}_"
                         f"{os.path.splitext(os.path.basename(best_image))[0]}_"
                         f"iou_{best_iou:.4f}.png")
            shutil.copyfile(best_visualization, os.path.join(best_dir, best_name))

        # Mean IoU for this query

        if len(query_ious) > 0:

            mean_iou = np.mean(query_ious)
            mean_macc = np.mean(query_maccs)

            print(
                f"\nMean IoU for '{query}' : {mean_iou:.4f}\n"
            )

        else:

            mean_iou = 0.0
            mean_macc = 0.0

            print(
                f"\nNo ground truth found for '{query}'.\n"
            )

        save_imagewise_results(
            query_output_dir,
            query,
            image_results
        )

        query_metrics[query] = float(mean_iou)
        query_macc_metrics[query] = float(mean_macc)

        global_metrics["per_query"][query] = float(mean_iou)
        global_metrics["per_query_mAcc"][query] = float(mean_macc)
        global_metrics["per_query_metrics"][query] = {
            "mean_iou": float(mean_iou),
            "mAcc": float(mean_macc),
            "frames": len(query_ious)
        }

        per_query_results.append({
            "method": "SceneSplat",
            "scene": scene_name,
            "query": query,
            "similarity_method": (
                "THGS-style query-vs-canonical"
            ),
            "selection_mode": (
                "fixed_similarity_threshold_with_knn_voting"
            ),
            "neighbor_voting": True,
            "vote_k": vote_k,

            "threshold_source": threshold_source,


            "mean_iou": float(mean_iou),
            "mAcc": float(mean_macc),

            "threshold": float(
                threshold
            ),

            "raw_selected_gaussians": int(
                len(selected_indices)
            ),

            "voted_selected_gaussians": int(
                len(selected_indices_np)
            )
        })


        # Save retrieval and visualization files

        preview_selected_gaussians(
            selected_indices_voted,
            selected_scores_voted
        )

        total_gaussians = features.shape[0]

        retrieval_file = save_retrieval_results(
            query_output_dir,
            query,
            selected_indices_voted,
            selected_scores_voted,
            threshold,
            total_gaussians,
            threshold_source
        )

        mask = build_visualization_mask(
            total_gaussians,
            selected_indices_voted
        )


        colors = build_visualization_colors(
            total_gaussians,
            selected_indices_voted
        )

        color_file = save_visualization_colors(
            query_output_dir,
            query,
            colors
        )

        mask_file = save_visualization_mask(
            query_output_dir,
            query,
            mask
        )

        print("\n---------- Query Summary ----------")
        print(f"Query                : {query}")
        print(
            "Selection method     : "
            "Fixed similarity threshold + kNN voting"
        )

        print(
            f"Threshold source     : "
            f"{threshold_source}"
        )

        print(
            f"Similarity threshold : "
            f"{threshold:.6f}"
        )

        print(
            f"Mean IoU             : "
            f"{mean_iou:.4f}"
        )

        print(
            f"Raw Gaussians        : "
            f"{len(selected_indices)}"
        )

        print(
            f"Voted Gaussians      : "
            f"{len(selected_indices_np)}"
        )

        retrieval_files.append(retrieval_file)
        mask_files.append(mask_file)
        color_files.append(color_file)
    
    # Global Evaluation

    metrics = compute_global_metrics(
        query_metrics,
        query_macc_metrics
    )

    global_metrics.update(metrics)
    global_metrics["frame_query_mIoU"] = float(np.mean(all_frame_ious))
    global_metrics["frame_query_mAcc"] = float(np.mean(all_frame_maccs))

    save_global_metrics(
        scene_output_dir,
        global_metrics
    )

    save_per_query_csv(
        scene_output_dir,
        per_query_results
    )

    save_evaluation_summary(
        scene_output_dir,
        queries_to_evaluate,
        retrieval_files,
        mask_files,
        color_files
    )
    save_coloured_mask_results(scene_output_dir, args.image_root, coloured_entries)

    
    print_evaluation_statistics(
        queries_to_evaluate,
        features.shape[0],
        retrieval_files,
        mask_files,
        color_files
    )

    print("\n========== Overall Evaluation ==========\n")

    print(f"Method              : {global_metrics['method']}")
    print(f"Scene               : {global_metrics['scene']}")

    print()

    print(f"Queries evaluated   : {metrics['queries']}")

    print(f"Mean IoU            : {metrics['mean_iou']:.4f}")
    print(f"Median IoU          : {metrics['median_iou']:.4f}")
    print(f"Std IoU             : {metrics['std_iou']:.4f}")

    print(f"Min IoU             : {metrics['min_iou']:.4f}")
    print(f"Max IoU             : {metrics['max_iou']:.4f}")

    print()

    print(f"Precision@0.25      : {metrics['precision@0.25']:.4f}")
    print(f"Precision@0.50      : {metrics['precision@0.50']:.4f}")

    print("\n=========================================\n")


    print(f"Scene folder        : {args.input_root}")

    print("\n========== Scene Summary ==========\n")
    print(f"Scene name          : {scene_name}")
    print(f"Number of Gaussians : {scene_data['coord'].shape[0]}")

    print(f"Loaded arrays       : {sorted(scene_data.keys())}")
   
    print(f"Annotation folder   : {args.label_root}")
    print(f"COLMAP folder       : {args.colmap_sparse}")
    print(f"Ground truth images : {len(annotations)}")
    print(f"Unique categories   : {len(categories)}")
    

    if isinstance(features, torch.Tensor):

        print(f"Feature shape       : {tuple(features.shape)}")
        print(f"Feature dtype       : {features.dtype}")

    elif isinstance(features, dict):

        print("\nFeature tensors:")

        for key, value in features.items():

            if isinstance(value, torch.Tensor):

                print(
                    f"  {key:<18}: {tuple(value.shape)}"
                )

    print(
        "Similarity method   : "
        "THGS-style query-vs-canonical"
    )

    print(
        f"Canonical prompts   : "
        f"{CANONICAL_PROMPTS}"
    )
    print(f"Retrieval files     : {len(retrieval_files)}")
    print(f"Scene Gaussians     : {features.shape[0]}")
    print(f"Annotation images   : {len(annotations)}")
    print(f"Visualization masks : {len(mask_files)}")
    print(f"Visualization colors : {len(color_files)}")


    print("\n===================================\n")

    print("[INFO] Evaluation completed successfully.")


def main():
    args = parse_args()
    output_root = args.output_dir
    shared = {}
    for threshold in args.thresholds:
        output_dir = os.path.join(
            output_root, "lerf_new_runs", f"results_{round(threshold * 100):02d}"
        )
        print(f"[THRESHOLD] Running fixed value {threshold:.2f}: {output_dir}", flush=True)
        run_evaluation(args, threshold, output_dir, shared)


if __name__ == "__main__":
    main()
