"""
Visualize SceneSplat ScanNet200 predictions.

"""

import os
import argparse
from pathlib import Path


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")   # headless rendering

import matplotlib.pyplot as plt
from PIL import Image

from matplotlib.colors import ListedColormap
from tqdm import tqdm


###############################################################################
# Paths
###############################################################################

DATA_ROOT = Path("data/preprocessedscene/scannet/val")
PRED_ROOT = Path("outputs/scannet_eval/result_ScanNet200GSDataset")
OUTPUT_ROOT = Path("outputs/scannet_eval/scannet_visualizations")


###############################################################################
# Rendering Settings
###############################################################################

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900

BACKGROUND_COLOR = [1, 1, 1, 1]

POINT_SIZE = 2.5


###############################################################################
# ScanNet ignored label
###############################################################################

IGNORE_LABEL = -1


###############################################################################
# Utility Functions
###############################################################################

def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def normalize_rgb(color):
    """
    Convert uint8 RGB to float RGB.
    """

    color = color.astype(np.float32)

    if color.max() > 1:
        color /= 255.0

    return color


def random_color(seed):

    rng = np.random.default_rng(seed)

    return rng.random(3)


def label_to_color(labels, color_map):
    """
    Convert semantic labels into RGB colors.

    labels:
        (N,)

    returns

        (N,3)
    """

    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)

    for label in np.unique(labels):

        if label == IGNORE_LABEL:

            colors[labels == label] = np.array([0.2, 0.2, 0.2])

            continue

        if label < len(color_map):

            colors[labels == label] = color_map[label]

        else:

            colors[labels == label] = random_color(label)

    return colors


def create_overlay(rgb, semantic_colors, alpha=0.45):
    """
    Blend RGB with semantic colors.

    overlay = alpha*semantic + (1-alpha)*rgb
    """

    rgb = normalize_rgb(rgb)

    overlay = alpha * semantic_colors + (1 - alpha) * rgb

    overlay = np.clip(overlay, 0, 1)

    return overlay


###############################################################################
# Open3D Helper
###############################################################################

def create_point_cloud(points, colors):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


###############################################################################
# Figure Helper
###############################################################################

def save_merged(rgb, gt, pred, overlay, save_path):

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("RGB")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(gt)
    ax[0, 1].set_title("Ground Truth")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(pred)
    ax[1, 0].set_title("Prediction")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(overlay)
    ax[1, 1].set_title("Overlay")
    ax[1, 1].axis("off")

    plt.tight_layout()

    plt.savefig(save_path, dpi=200)

    plt.close()

###############################################################################
# Color Map
###############################################################################

def generate_scannet200_colormap(num_classes=200):
    """
    Generate a deterministic color map for ScanNet200 classes.

    Colors are stable across runs and visually distinct.
    """

    rng = np.random.default_rng(42)

    colors = rng.random((num_classes, 3))

    # Make first few colors more distinctive
    colors[0] = [0.0, 0.0, 0.0]
    colors[1] = [1.0, 0.0, 0.0]
    colors[2] = [0.0, 1.0, 0.0]
    colors[3] = [0.0, 0.0, 1.0]
    colors[4] = [1.0, 1.0, 0.0]
    colors[5] = [1.0, 0.0, 1.0]
    colors[6] = [0.0, 1.0, 1.0]

    return colors


COLOR_MAP = generate_scannet200_colormap()

###############################################################################
# Utility Functions
###############################################################################

def colorize_labels(labels, color_map):
    """
    Convert integer semantic labels into RGB colors.
    Unknown labels are colored gray.
    """

    labels = labels.astype(np.int64)

    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)

    valid = (labels >= 0) & (labels < len(color_map))

    colors[valid] = color_map[labels[valid]]

    colors[~valid] = np.array([0.5, 0.5, 0.5])

    return colors


def overlay_prediction(rgb, gt, pred,
                       alpha_correct=0.70,
                       alpha_wrong=0.90):
    """
    Overlay prediction on RGB.

    Correct prediction:
        Mostly RGB.

    Wrong prediction:
        Mostly prediction color.
    """

    pred_color = colorize_labels(pred, COLOR_MAP)

    rgb = rgb.astype(np.float32)

    overlay = rgb.copy()

    correct = (gt == pred)

    overlay[correct] = (
        alpha_correct * rgb[correct]
        + (1.0 - alpha_correct) * pred_color[correct]
    )

    overlay[~correct] = (
        (1.0 - alpha_wrong) * rgb[~correct]
        + alpha_wrong * pred_color[~correct]
    )

    return np.clip(overlay, 0, 1)


def create_point_cloud(points, colors):
    """
    Build Open3D point cloud.
    """

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def save_point_cloud(path, points, colors):
    """
    Save colored point cloud.
    """

    pcd = create_point_cloud(points, colors)

    o3d.io.write_point_cloud(path, pcd)

    print(f"Saved: {path}")


def save_scene_outputs(
    scene,
    coord,
    rgb,
    gt,
    pred,
    output_root,
):
    """
    Save RGB / GT / Prediction / Overlay point clouds.
    """

    scene_dir = output_root / scene

    scene_dir.mkdir(parents=True, exist_ok=True)

    gt_color = colorize_labels(gt, COLOR_MAP)

    pred_color = colorize_labels(pred, COLOR_MAP)

    overlay = overlay_prediction(rgb, gt, pred)

    save_point_cloud(scene_dir / "rgb.ply", coord, rgb)

    save_point_cloud(scene_dir / "gt.ply", coord, gt_color)

    save_point_cloud(scene_dir / "prediction.ply", coord, pred_color)

    save_point_cloud(scene_dir / "overlay.ply", coord, overlay)

    return {
        "rgb": scene_dir / "rgb.ply",
        "gt": scene_dir / "gt.ply",
        "prediction": scene_dir / "prediction.ply",
        "overlay": scene_dir / "overlay.ply",
    }

###############################################################################
# Main Processing
###############################################################################

def process_scene(scene_name, args):
    print("=" * 80)
    print(f"Processing {scene_name}")
    print("=" * 80)

    scene_dir = args.data_root / scene_name

    pred_file = args.pred_root / f"{scene_name}_pred.npy"

    if not pred_file.exists():
        print(f"Prediction not found: {pred_file}")
        return

    coord = np.load(scene_dir / "coord.npy")
    color = np.load(scene_dir / "color.npy")
    gt = np.load(scene_dir / "segment200.npy")
    pred = np.load(pred_file)

    color = color.astype(np.float32)

    if color.max() > 1:
        color /= 255.0

    if len(coord) != len(pred):
        print("Point count mismatch!")
        print(coord.shape)
        print(pred.shape)
        return

    save_scene_outputs(
        scene=scene_name,
        coord=coord,
        rgb=color,
        gt=gt,
        pred=pred,
        output_root=args.output_root,
    )


    generate_png_outputs(
        scene=scene_name,
        coord=coord,
        rgb=color,
        gt=gt,
        pred=pred,
        output_root=args.output_root,
    )


    print(f"Finished {scene_name}\n")

###############################################################################
# Rendering the PNGs
###############################################################################
def render_png(points,
               colors,
               save_path,
               sample_size=150000,
               elev=25,
               azim=-60):
    """
    Render a colored point cloud into a PNG using matplotlib.
    """

    n = len(points)

    if n > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, sample_size, replace=False)

        points = points[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        points[:,0],
        points[:,1],
        points[:,2],
        c=colors,
        s=0.2,
        marker='.'
    )

    ax.view_init(elev=elev, azim=azim)

    ax.set_axis_off()

    ax.set_box_aspect([
        np.ptp(points[:,0]),
        np.ptp(points[:,1]),
        np.ptp(points[:,2])
    ])

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    print(f"Saved: {save_path}")


    ################################################
    # Generate PNGs
    ################################################
def generate_png_outputs(
        scene,
        coord,
        rgb,
        gt,
        pred,
        output_root
):
    """
    Generate PNG visualizations from point clouds.
    """

    scene_dir = output_root / scene

    png_dir = scene_dir / "png"

    png_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Colors
    gt_color = colorize_labels(
        gt,
        COLOR_MAP
    )

    pred_color = colorize_labels(
        pred,
        COLOR_MAP
    )

    overlay = overlay_prediction(
        rgb,
        gt,
        pred
    )


    ################################################
    # Individual PNGs
    ################################################

    render_png(
        coord,
        rgb,
        png_dir / "rgb.png"
    )


    render_png(
        coord,
        gt_color,
        png_dir / "ground_truth.png"
    )


    render_png(
        coord,
        pred_color,
        png_dir / "prediction.png"
    )


    render_png(
        coord,
        overlay,
        png_dir / "overlay.png"
    )



    ################################################
    # Merged 2x2 figure
    ################################################

    from mpl_toolkits.mplot3d import Axes3D


    fig = plt.figure(figsize=(12,10))


    titles = [
        "RGB",
        "Ground Truth",
        "Prediction",
        "Overlay"
    ]


    colors_list = [
        rgb,
        gt_color,
        pred_color,
        overlay
    ]


    for i, (title, colors) in enumerate(
            zip(titles, colors_list)
    ):

        ax = fig.add_subplot(
            2,
            2,
            i+1,
            projection="3d"
        )


        n = len(coord)

        if n > 150000:

            rng = np.random.default_rng(42)

            idx = rng.choice(
                n,
                150000,
                replace=False
            )

            pts = coord[idx]
            cols = colors[idx]

        else:

            pts = coord
            cols = colors


        ax.scatter(
            pts[:,0],
            pts[:,1],
            pts[:,2],
            c=cols,
            s=0.2
        )


        ax.set_title(title)

        ax.set_axis_off()


        ax.view_init(
            elev=25,
            azim=-60
        )


    plt.tight_layout()


    merged_path = png_dir / "merged.png"


    plt.savefig(
        merged_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)


    print(f"Saved merged PNG: {merged_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/preprocessedscene/scannet/val"),
    )

    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("outputs/scannet_eval/result_ScanNet200GSDataset"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/scannet_eval/scannet_visualizations"),
    )

    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Only visualize one scene.",
    )

    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.scene is not None:

        process_scene(args.scene, args)

    else:

        scenes = sorted(
            [
                p.name
                for p in args.data_root.iterdir()
                if p.is_dir()
            ]
        )

        print(f"Found {len(scenes)} scenes")

        for scene in scenes:
            process_scene(scene, args)

    print("\nDone.")


if __name__ == "__main__":
    main()