import os
import argparse
import glob
import numpy as np
import torch
import json
import csv
import cv2
import struct
import collections
import matplotlib.pyplot as plt

from PIL import Image as PILImage
from scripts.encode_labels import SigLIPLabelEncoder


# COLMAP Parser reuse from Gaussian Splatting repo

# ----------------------------------------------------------
# COLMAP Camera Structures
# ----------------------------------------------------------

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
# ----------------------------------------------------------
# Quaternion → Rotation Matrix
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Read Binary Bytes
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Read COLMAP Cameras
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Read COLMAP Images
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Load COLMAP Camera Data
# ----------------------------------------------------------

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



# ----------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="SceneSplat Evaluation on LERF"
    )

    parser.add_argument(
        "--config",
        default=None,
        help="SceneSplat inference config (used in later milestones)"
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
        help="Directory to save evaluation outputs"
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

    return parser.parse_args()

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


# ----------------------------------------------------------
# Load Preprocessed Scene
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Load SceneSplat Features
# ----------------------------------------------------------

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
# ----------------------------------------------------------
# Load LERF Annotations
# 
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
# ----------------------------------------------------------
# Build Text Embeddings
# ----------------------------------------------------------

def build_text_embeddings(annotations):
    """
    Encode every unique object category using SceneSplat's
    SigLIP2 text encoder.
    """

    print("\n[INFO] Building text embeddings...\n")

    # Collect unique categories
    categories = sorted({

        obj["category"]

        for image in annotations.values()

        for obj in image

    })

    print(f"Found {len(categories)} unique categories:\n")

    for category in categories:
        print(f"  - {category}")

    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SigLIPLabelEncoder(device=device)

    text_embeddings = encoder.encode_labels(categories)

    print("\n[INFO] Text embeddings created.")
    print(f"Embedding shape : {tuple(text_embeddings.shape)}")
    print(f"Embedding dtype : {text_embeddings.dtype}")

    return categories, text_embeddings


# ----------------------------------------------------------
# Normalize Text Embeddings
# ----------------------------------------------------------

def normalize_text_embeddings(text_embeddings):

    print("\n[INFO] Normalizing text embeddings...\n")

    text_embeddings = torch.nn.functional.normalize(
        text_embeddings.float(),
        dim=1
    )

    print(
        f"Normalized text embeddings : {tuple(text_embeddings.shape)}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return text_embeddings.to(device)

# ----------------------------------------------------------
# Normalize Scene Features
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Compute Gaussian-Text Similarity
# ----------------------------------------------------------

def compute_similarity(scene_features, text_embeddings):
    """
    Compute cosine similarity between every Gaussian feature
    and every text embedding.
    """

    print("\n[INFO] Computing Gaussian-Text similarity...\n")

    if isinstance(scene_features, dict):

        raise RuntimeError(
            "Current feature file is a dictionary. "
            "Need tensor features first."
        )

    scene_features = scene_features.float()
    text_embeddings = text_embeddings.float()

    if scene_features.device != text_embeddings.device:

        raise RuntimeError(
            f"Device mismatch: "
            f"{scene_features.device} vs {text_embeddings.device}"
        )

    similarity = scene_features @ text_embeddings.T

    print(f"Similarity matrix : {tuple(similarity.shape)}")

    return similarity

# ----------------------------------------------------------
# Predict Label for Every Gaussian
# ----------------------------------------------------------

def predict_gaussian_labels(similarity, categories):
    """
    Assign every Gaussian the category with the highest similarity.
    """

    print("\n[INFO] Predicting Gaussian labels...\n")

    best_scores, best_indices = similarity.max(dim=1)

    predicted_labels = [
        categories[idx]
        for idx in best_indices.tolist()
    ]

    print(f"Predicted labels : {len(predicted_labels)}")

    return predicted_labels, best_scores

# ----------------------------------------------------------
# Preview Predictions
# ----------------------------------------------------------

def preview_predictions(predicted_labels, scores, num_examples=10):

    print("\n========== Prediction Preview ==========\n")

    for i in range(min(num_examples, len(predicted_labels))):

        print(
            f"Gaussian {i:<6}"
            f"{predicted_labels[i]:<20}"
            f"{scores[i]:.4f}"
        )

    print("\n========================================\n")

# ----------------------------------------------------------
# Retrieve Gaussian Similarity for One Query
# ----------------------------------------------------------

def retrieve_query(
    query,
    categories,
    similarity
):
    """
    Returns similarity scores for a single object query.
    """

    if query not in categories:
        raise RuntimeError(
            f"{query} not found in category list."
        )

    query_index = categories.index(query)

    query_similarity = similarity[:, query_index]

    print(f"\nQuery : {query}")
    print(f"Retrieved {query_similarity.shape[0]} Gaussian scores")

    return query_similarity

# ----------------------------------------------------------
# Show Top Retrieval Results
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Save Gaussian Predictions
# ----------------------------------------------------------

def save_predictions(
    predicted_labels,
    scores,
    output_dir
):
    """
    Save predicted label and confidence
    for every Gaussian.
    """

    print("\n[INFO] Saving predictions...\n")

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "gaussian_predictions.json"
    )

    results = []

    for idx, (label, score) in enumerate(
        zip(predicted_labels, scores)
    ):

        results.append({
            "gaussian_id": idx,
            "label": label,
            "score": float(score)
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved predictions to:\n{output_file}\n")


# ----------------------------------------------------------
# Build Gaussian Heatmap
# ----------------------------------------------------------

def build_gaussian_heatmap(
    query,
    query_similarity
):
    """
    Creates a per-Gaussian heatmap for one query.

    Returns
    -------
    heatmap : Tensor
        One similarity score per Gaussian.
    """

    print(f"\n[INFO] Building heatmap for '{query}'...")

    heatmap = query_similarity.clone()

    print(
        f"Heatmap size : {tuple(heatmap.shape)}"
    )

    print(
        f"Similarity range : "
        f"{heatmap.min():.4f} "
        f"to "
        f"{heatmap.max():.4f}"
    )

    return heatmap

# ----------------------------------------------------------
# Threshold Gaussian Heatmap
# ----------------------------------------------------------

def threshold_heatmap(
    heatmap,
    threshold=0.30
):
    """
    Select Gaussians whose similarity is above a threshold.

    Returns
    -------
    selected_indices : Tensor
        Indices of selected Gaussians.

    selected_scores : Tensor
        Similarity scores of selected Gaussians.
    """

    print("\n[INFO] Thresholding heatmap...\n")

    mask = heatmap >= threshold

    selected_indices = torch.nonzero(
        mask,
        as_tuple=False
    ).squeeze(1)

    selected_scores = heatmap[selected_indices]

    print(f"Threshold          : {threshold:.2f}")
    print(f"Selected Gaussians : {len(selected_indices)}")

    return selected_indices, selected_scores


# ----------------------------------------------------------
# Adaptive Heatmap Threshold
# ----------------------------------------------------------

def adaptive_threshold_heatmap(
    heatmap,
    percentile=95
):
    """
    Keep only the top X percentile of Gaussians.

    Parameters
    ----------
    percentile : int

        Example:
            95 -> keep top 5%
            90 -> keep top 10%
    """

    print("\n[INFO] Adaptive thresholding...\n")

    threshold = torch.quantile(
        heatmap,
        percentile / 100.0
    )

    mask = heatmap >= threshold

    selected_indices = torch.nonzero(
        mask,
        as_tuple=False
    ).squeeze(1)

    selected_scores = heatmap[selected_indices]

    print(f"Percentile         : {percentile}")
    print(f"Threshold          : {threshold:.4f}")
    print(f"Selected Gaussians : {len(selected_indices)}")

    return (
        selected_indices,
        selected_scores,
        threshold
    )


# ----------------------------------------------------------
# Preview Selected Gaussians
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Save Retrieval Results
# ----------------------------------------------------------

def save_retrieval_results(
    output_dir,
    query,
    selected_indices,
    selected_scores,
    threshold,
    total_gaussians
):
    """
    Save retrieval results for one query as JSON.
    """

    print("\n[INFO] Saving retrieval results...\n")

    os.makedirs(output_dir, exist_ok=True)
    selected_fraction = len(selected_indices) / total_gaussians
    result = {
        "query": query,
        "threshold_type": "adaptive_percentile",
        "threshold": float(threshold),
        "num_selected": int(len(selected_indices)),
        "selected_fraction": float(selected_fraction),
        "gaussian_indices": selected_indices.cpu().tolist(),
        "similarity_scores": [
            float(score)
            for score in selected_scores.cpu()
        ]
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

# ----------------------------------------------------------
# Build Visualization Mask
# ----------------------------------------------------------

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

    mask[selected_indices] = 1

    return mask

# ----------------------------------------------------------
# Save Visualization Mask
# ----------------------------------------------------------

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
# ----------------------------------------------------------
# Build Gaussian Visualization Colors
# ----------------------------------------------------------

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

    colors[selected_indices.cpu().numpy()] = np.array(
        [255, 0, 0],
        dtype=np.uint8
    )

    return colors

# ----------------------------------------------------------
# Project Gaussians Function
# ----------------------------------------------------------
def project_gaussians(
    xyz,
    images,
    cameras,
    image_name
):
    """
    Project Gaussian centers into one image.
    """

    # -----------------------------------
    # Find COLMAP image
    # -----------------------------------

    image = None

    for img in images.values():

        if img.name == image_name:
            image = img
            break

    if image is None:
        raise RuntimeError(f"{image_name} not found.")

    camera = cameras[image.camera_id]

    # -----------------------------------
    # Camera intrinsics
    # -----------------------------------

    fx = camera.params[0]
    fy = camera.params[1]
    cx = camera.params[2]
    cy = camera.params[3]

    width = camera.width
    height = camera.height

    # -----------------------------------
    # World → Camera
    # -----------------------------------

    R = image.qvec2rotmat()
    t = image.tvec

    xyz_cam = xyz @ R.T + t

    x = xyz_cam[:,0]
    y = xyz_cam[:,1]
    z = xyz_cam[:,2]

    # -----------------------------------
    # Perspective projection
    # -----------------------------------

    u = fx * x / z + cx
    v = fy * y / z + cy

    pixels = np.stack([u, v], axis=1)

    # -----------------------------------
    # Valid points
    # -----------------------------------

    valid = (
        (z > 0) &
        (u >= 0) &
        (u < width) &
        (v >= 0) &
        (v < height)
    )

    return pixels, valid, z



def render_prediction_mask(
    projected_pixels,
    depths,
    valid_mask,
    selected_indices,
    scales,
    image_shape,
    radius_scale=80.0,
    min_radius=2,
    max_radius=12
):
    """
    Render selected Gaussians using

    • depth buffering
    • Gaussian radius
    • filled circles
    """

    H, W = image_shape

    prediction = np.zeros((H, W), dtype=np.uint8)

    depth_buffer = np.full((H, W), np.inf)

    selected = np.zeros(len(valid_mask), dtype=bool)
    selected[selected_indices.cpu().numpy()] = True

    final = valid_mask & selected

    indices = np.where(final)[0]

    # nearest Gaussian first
    indices = indices[np.argsort(depths[indices])]

    for idx in indices:

        x = int(round(projected_pixels[idx, 0]))
        y = int(round(projected_pixels[idx, 1]))

        if not (0 <= x < W and 0 <= y < H):
            continue

        if depths[idx] > depth_buffer[y, x]:
            continue

        depth_buffer[y, x] = depths[idx]

        sigma = np.mean(scales[idx])

        radius = int(radius_scale * sigma)

        radius = np.clip(
            radius,
            min_radius,
            max_radius
        )

        cv2.circle(
            prediction,
            (x, y),
            radius,
            1,
            -1
        )

    return prediction


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
    
# ----------------------------------------------------------
# Save Visualization Colors
# ----------------------------------------------------------

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
    image_name,
    save_best=False
):
    """
    Save visualization comparing

    1. RGB image
    2. Ground-truth mask
    3. Predicted mask
    4. Overlay
    """


    folder = (
        "best_visualizations"
        if save_best
        else "mask_visualizations"
    )

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

    axes[1].imshow(gt_mask,cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction_mask,cmap="gray")
    axes[2].set_title("Prediction")
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
# ----------------------------------------------------------
# Save Evaluation Summary
# ----------------------------------------------------------

def save_evaluation_summary(
    output_dir,
    categories,
    retrieval_files,
    mask_files,
    color_files,
    evaluation_files
):
    """
    Save overall evaluation summary.
    """

    summary = {
        "num_categories": len(categories),
        "categories": categories,
        "retrieval_files": retrieval_files,
        "mask_files": mask_files,
        "color_files": color_files,
        "evaluation_files": evaluation_files
    }

    filename = os.path.join(
        output_dir,
        "evaluation_summary.json"
    )

    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved evaluation summary : {filename}")

# ----------------------------------------------------------
# Save Per Query CSV
# ----------------------------------------------------------

# ----------------------------------------------------------
# Save Per Query CSV
# ----------------------------------------------------------

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
            "Threshold",
            "Selected_Gaussians"
        ])

        for row in per_query_results:

            writer.writerow([
                row["method"],
                row["scene"],
                row["query"],
                round(row["mean_iou"], 6),
                round(row["threshold"], 6),
                row["selected_gaussians"]
            ])

    print(f"Saved CSV : {filename}")

    return filename

# ----------------------------------------------------------
# Save Image-wise IoU Results
# ----------------------------------------------------------

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

        writer.writerow([
            "Image",
            "IoU"
        ])

        for image_name, iou in image_results:

            writer.writerow([
                image_name,
                round(iou, 6)
            ])

    print(f"Saved image-wise results : {filename}")

    return filename


# ----------------------------------------------------------
# Compute Global Evaluation Metrics
# ----------------------------------------------------------

def compute_global_metrics(
    query_metrics
):

    values = np.array(
        list(query_metrics.values()),
        dtype=np.float32
    )

    metrics = {}

    metrics["queries"] = len(values)

    metrics["mean_iou"] = float(np.mean(values))

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

# ----------------------------------------------------------
# Save Global Metrics
# ----------------------------------------------------------

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

    return filename



# ----------------------------------------------------------
# Inspect first LERF Annotations
# 
def inspect_first_annotation(annotations):

    first_image = next(iter(annotations))

    print(f"Image : {first_image}\n")

    for obj in annotations[first_image]:

        print(
            f"{obj['category']:<25}"
            f"1 polygon"
        )

    print("\n=====================================\n")


# ----------------------------------------------------------
# Print Evaluation Statistics
# ----------------------------------------------------------

def print_evaluation_statistics(
    categories,
    predicted_labels,
    retrieval_files,
    mask_files,
    color_files,
    evaluation_files
):

    print("\n========== Evaluation Statistics ==========\n")

    print(f"Queries evaluated      : {len(categories)}")
    print(f"Predicted Gaussians    : {len(predicted_labels)}")
    print(f"Retrieval files        : {len(retrieval_files)}")
    print(f"Visualization masks    : {len(mask_files)}")
    print(f"Visualization colors   : {len(color_files)}")
    print(f"Evaluation files       : {len(evaluation_files)}")

    print("\n===========================================\n")



# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    args = parse_args()

    scene_name = os.path.basename(
            os.path.normpath(args.input_root)
        )
    
    scene_data = load_preprocessed_scene(
        args.input_root
    )

    features = load_scenesplat_features(
        args.feature_path
    )
    query_metrics = {}
    per_query_results = []

    global_metrics = {
        "scene": os.path.basename(
            os.path.normpath(args.input_root)
        ),
        "method": "SceneSplat",
        "per_query": {}
    }


    annotations = load_lerf_annotations(args.label_root)

    cameras, images = load_camera_data(args.colmap_sparse)

    inspect_first_annotation(annotations)

    categories, text_embeddings = build_text_embeddings(
    annotations
    )


    text_embeddings = normalize_text_embeddings(
    text_embeddings
    )

    features = normalize_scene_features(features)

    similarity = compute_similarity(
        features,
        text_embeddings
    )
    
    predicted_labels, prediction_scores = predict_gaussian_labels(
    similarity,
    categories
    )

    preview_predictions(
    predicted_labels,
    prediction_scores
    )

    save_predictions(
            predicted_labels,
            prediction_scores,
            args.output_dir
        )
    # mask_vis_dir = output_dir / "mask_visualizations"
    # mask_vis_dir.mkdir(parents=True, exist_ok=True)

    # best_vis_dir = output_dir / "best_visualizations"
    # best_vis_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== Open Vocabulary Retrieval ==========\n")

    retrieval_files = []
    mask_files = []
    color_files = []
    evaluation_files = []

    for query in categories:

        print(f"\nProcessing query : {query}")

        query_similarity = retrieve_query(
            query,
            categories,
            similarity
        )

        show_top_retrievals(
            query,
            query_similarity
        )

        heatmap = build_gaussian_heatmap(
            query,
            query_similarity
        )

        selected_indices, selected_scores, threshold = \
            adaptive_threshold_heatmap(
                heatmap,
                percentile=95
            )
        query_ious = []
        image_results = []
        best_iou = -1.0
        best_image = None
        best_gt_mask = None
        best_prediction_mask = None
        best_image_name = None

        # =====================================================
        # Evaluate this query on every annotated image
        # =====================================================

        for image_name, annotation in annotations.items():

            print(f"\nEvaluating image : {image_name}")

            # ---------------------------------------------
            # Load RGB image
            # ---------------------------------------------

            image_path = os.path.join(
                args.image_root,
                image_name
            )

            image = load_camera_image(image_path)

            # ---------------------------------------------
            # Build Ground Truth Masks
            # ---------------------------------------------

            gt_masks = build_ground_truth_masks(
                annotation,
                image
            )

            # Skip images that do not contain this object
            if query not in gt_masks:
                continue

            gt_mask = gt_masks[query]

            # ---------------------------------------------
            # Project Gaussians
            # ---------------------------------------------

            projected_pixels, valid_mask, depths = project_gaussians(
                scene_data["coord"],
                images,
                cameras,
                image_name
            )

            # ---------------------------------------------
            # Render prediction mask
            # ---------------------------------------------

            prediction_mask = render_prediction_mask(
                projected_pixels,
                depths,
                valid_mask,
                selected_indices,
                scene_data["scale"],
                image.shape[:2]
            )

            save_mask_visualization(
                image=image,
                gt_mask=gt_mask,
                prediction_mask=prediction_mask,
                output_dir=args.output_dir,
                query=query,
                image_name=os.path.splitext(image_name)[0],
                save_best=False
            )

            # ---------------------------------------------
            # Evaluate IoU
            # ---------------------------------------------

            iou = evaluate_iou(
                prediction_mask,
                gt_mask
            )
            if iou > best_iou:
                best_iou = iou
                best_image = image.copy()
                best_gt_mask = gt_mask.copy()
                best_prediction_mask = prediction_mask.copy()
                best_image_name = image_name
                
            query_ious.append(iou)


            image_results.append(
                (
                    image_name,
                    float(iou)
                )
            )

            print(
                f"{image_name:<25} IoU : {iou:.4f}"
            )

        # =====================================================
        # Mean IoU for this query
        # =====================================================

        if len(query_ious) > 0:

            mean_iou = np.mean(query_ious)

            print(
                f"\nMean IoU for '{query}' : {mean_iou:.4f}\n"
            )

        else:

            mean_iou = 0.0

            print(
                f"\nNo ground truth found for '{query}'.\n"
            )

        # ---------------------------------------------------------
        # Save ONLY the best IoU visualization for this query
        # ---------------------------------------------------------
        if best_image is not None:

            save_mask_visualization(
                image=best_image,
                gt_mask=best_gt_mask,
                prediction_mask=best_prediction_mask,
                output_dir=args.output_dir,
                query=query,
                image_name=os.path.splitext(best_image_name)[0],
                save_best=True,
            )

        save_imagewise_results(
            args.output_dir,
            query,
            image_results
        )

        query_metrics[query] = float(mean_iou)

        global_metrics["per_query"][query] = float(mean_iou)

        per_query_results.append({

            "method": "SceneSplat",

            "scene": scene_name,

            "query": query,

            "mean_iou": float(mean_iou),

            "threshold": float(threshold),

            "selected_gaussians": int(len(selected_indices))

        })



        # =====================================================
        # Save retrieval and visualization files
        # =====================================================

        preview_selected_gaussians(
            selected_indices,
            selected_scores
        )

        retrieval_file = save_retrieval_results(
            args.output_dir,
            query,
            selected_indices,
            selected_scores,
            threshold,
            similarity.shape[0]
        )

        mask = build_visualization_mask(
            similarity.shape[0],
            selected_indices
        )

        colors = build_visualization_colors(
            similarity.shape[0],
            selected_indices
        )

        color_file = save_visualization_colors(
            args.output_dir,
            query,
            colors
        )

        mask_file = save_visualization_mask(
            args.output_dir,
            query,
            mask
        )

        print("\n---------- Query Summary ----------")
        print(f"Query                : {query}")
        print(f"Mean IoU             : {mean_iou:.4f}")
        print(f"Heatmap size         : {tuple(heatmap.shape)}")
        print(f"Adaptive threshold   : {threshold:.4f}")
        print(f"Selected Gaussians   : {len(selected_indices)}")
        print(f"Retrieval file       : {retrieval_file}")
        print("-----------------------------------\n")

        retrieval_files.append(retrieval_file)
        mask_files.append(mask_file)
        color_files.append(color_file)
    
    ############################################################
    # Global Evaluation
    ############################################################

    metrics = compute_global_metrics(
        query_metrics
    )

    global_metrics.update(metrics)

    save_global_metrics(
        args.output_dir,
        global_metrics
    )

    save_per_query_csv(
        args.output_dir,
        per_query_results
    )

    save_evaluation_summary(
        args.output_dir,
        categories,
        retrieval_files,
        mask_files,
        color_files,
        evaluation_files
    )

    required = [
        "coord",
        "color",
        "opacity",
        "quat",
        "scale"
    ]

    missing = [
        k for k in required
        if k not in scene_data
    ]

    if missing:

        raise RuntimeError(
            f"Missing required arrays: {missing}"
        )
    

    print_evaluation_statistics(
        categories,
        predicted_labels,
        retrieval_files,
        mask_files,
        color_files,
        evaluation_files
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
    f"Text embedding size : {tuple(text_embeddings.shape)}"
    )
    print(f"Similarity matrix   : {tuple(similarity.shape)}")
    print(f"Retrieval files     : {len(retrieval_files)}")
    print(f"Predicted labels    : {len(predicted_labels)}")
    print(f"Annotation images   : {len(annotations)}")
    print(f"Visualization masks : {len(mask_files)}")
    print(f"Visualization colors : {len(color_files)}")

    first_image = next(iter(annotations))

    print("\n===================================\n")

    print("[INFO] Evaluation completed successfully.")


if __name__ == "__main__":
    main()