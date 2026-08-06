import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from plyfile import PlyData
from collections import Counter, defaultdict
# ----------------------------------------------------------
# Reuse SceneSplat evaluation functions
# ----------------------------------------------------------

from tools.scenesplat_lerf_eval import (
    build_text_embeddings,
    normalize_text_embeddings,
    normalize_scene_features,
    compute_similarity,
    predict_gaussian_labels,
    project_gaussians,
    render_prediction_mask,
    evaluate_iou,
    load_camera_data,
    save_per_query_csv,
    save_imagewise_results,
    compute_global_metrics,
    save_global_metrics,
    load_camera_image,
    save_mask_visualization
)

# ----------------------------------------------------------
# Argument parser
# ----------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate SceneSplat on the 3DOVS dataset"
    )

    parser.add_argument(
        "--scene-root",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--feature-root",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output",
        default="results/3dovs_eval",
        type=str,
    )

    return parser.parse_args()

def load_feature_tensor(feature_root, scene_name):

    feature_file = (
        Path(feature_root)
        / scene_name
        / f"{scene_name}_feat.pt"
    )
    if not feature_file.exists():
        raise FileNotFoundError(feature_file)
    
    features = torch.load(
        feature_file,
        map_location="cpu"
    )

    print(
        f"{scene_name:<15}"
        f"{tuple(features.shape)}"
    )

    return features

def load_gaussian_data(scene_root, scene_name):
    """
    Load Gaussian positions and scales from gsplat PLY.
    """

    ply_file = (
        Path(scene_root)
        / scene_name
        / f"{scene_name}_gaussians.ply"
    )

    if not ply_file.exists():
        raise FileNotFoundError(ply_file)

    ply = PlyData.read(
        str(ply_file)
    )

    vertices = ply["vertex"]

    xyz = np.stack(
        [
            vertices["x"],
            vertices["y"],
            vertices["z"],
        ],
        axis=1,
    )

    scales = np.exp(
        np.stack(
            [
                vertices["scale_0"],
                vertices["scale_1"],
                vertices["scale_2"],
            ],
            axis=1,
        )
    )

    xyz = torch.from_numpy(xyz).float()

    scales = torch.from_numpy(scales).float()


    print(
        f"Gaussian XYZ: {tuple(xyz.shape)}"
    )

    print(
        f"Gaussian scales: {tuple(scales.shape)}"
    )


    return xyz, scales




def load_classes(dataset_root, scene_name):

    class_file = (
        Path(dataset_root)
        / scene_name
        / "segmentations"
        / "classes.txt"
    )

    if not class_file.exists():
        raise FileNotFoundError(class_file)

    with open(class_file) as f:

        classes = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(f"\nLoaded {len(classes)} classes:")
    for c in classes:
        print(f"  - {c}")

    return classes

def load_segmentation_masks(dataset_root, scene_name):
    """
    Discover all annotated images and their object masks.

    Returns
    -------
    {
        "00": {
            "banana": Path(...),
            "camera": Path(...),
            ...
        },
        "04": {
            ...
        }
    }
    """

    segmentation_root = (
        Path(dataset_root)
        / scene_name
        / "segmentations"
    )

    image_masks = {}

    for image_dir in sorted(segmentation_root.iterdir()):

        if not image_dir.is_dir():
            continue

        masks = {}

        for mask_file in sorted(image_dir.glob("*.png")):

            object_name = mask_file.stem

            masks[object_name] = mask_file

        image_masks[image_dir.name] = masks

    return image_masks


def verify_annotation_images(images, image_masks):
    """
    Verify annotated images exist in COLMAP.
    """

    image_lookup = {
        image.name
        for image in images.values()
    }

    print("\nChecking annotated images:\n")

    for image_name in sorted(image_masks):

        jpg = f"{image_name}.jpg"

        if jpg in image_lookup:

            print(f"  ✓ {jpg}")

        else:

            print(f"  ✗ {jpg}")



def build_text_embeddings_from_classes(classes):
    """
    Reuse the existing LERF text embedding pipeline by
    constructing the expected annotation format.
    """

    annotations = {
        "dummy": [
            {"category": c}
            for c in classes
        ]
    }

    return build_text_embeddings(annotations)

# def test_retrieval(categories, similarity):
#     """
#     Test retrieval for every object category.
#     """

#     print("\n================ Retrieval Test ================\n")

#     for query in categories:

#         query_similarity = retrieve_query(
#             query,
#             categories,
#             similarity,
#         )

#         print(
#             f"{query:<30}"
#             f"{tuple(query_similarity.shape)}"
#         )



#     print("\n===============================================\n")



def main():

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    scene_root = Path(args.scene_root)

    scenes = sorted(
        d.name
        for d in scene_root.iterdir()
        if d.is_dir()
    )

    print(f"Found {len(scenes)} scenes.\n")

    for scene in scenes:

        scene_results = []
        object_results = defaultdict(list)
        query_metrics = {}

        print("=" * 60)
        print(scene)
        scene_output = output_dir / scene
        scene_output.mkdir(
            parents=True,
            exist_ok=True
        )

        features = load_feature_tensor(
            args.feature_root,
            scene,
        )

        gaussian_xyz, gaussian_scales = load_gaussian_data(
            args.scene_root,
            scene,
        )

        assert gaussian_xyz.shape[0] == features.shape[0], \
            f"Mismatch: Gaussian {gaussian_xyz.shape[0]} vs Feature {features.shape[0]}"

        assert gaussian_scales.shape[0] == features.shape[0], \
            f"Mismatch: Scale {gaussian_scales.shape[0]} vs Feature {features.shape[0]}"

        classes = load_classes(
                    args.dataset_root,
                    scene,
                )
        
        image_masks = load_segmentation_masks(
            args.dataset_root,
            scene,
        )

        colmap_folder = (
            Path(args.dataset_root)
            / scene
            / "sparse"
            / "0"
        )

        cameras, images = load_camera_data(
            str(colmap_folder)
        )

        

        verify_annotation_images(
            images,
            image_masks,
        )

        print("\nAnnotated images:")

        for image_name, masks in image_masks.items():

            print(f"\nImage {image_name}")

            for object_name in masks:

                print(f"   - {object_name}")


        categories, text_embeddings = (
            build_text_embeddings_from_classes(
                classes
            )
        )

        text_embeddings = normalize_text_embeddings(
            text_embeddings
        )

        features = normalize_scene_features(
            features
        )

        # ----------------------------------------------------------
        # Move scene features to same device as text embeddings
        # ----------------------------------------------------------

        device = text_embeddings.device
        features = features.to(device)

        similarity = compute_similarity(
            features,
            text_embeddings,
        )

        print(f"Similarity matrix : {tuple(similarity.shape)}")


        predicted_labels, _ = predict_gaussian_labels(
            similarity,
            categories,
        )

        assert len(predicted_labels) == gaussian_xyz.shape[0], \
            f"Labels {len(predicted_labels)} != Gaussians {gaussian_xyz.shape[0]}"

        print(
            f"Number of predicted labels: {len(predicted_labels)}"
        )

        print(
            f"Example labels: {predicted_labels[:10]}"
        )

        # ----------------------------------------------------------
        # Evaluate each object category
        # ----------------------------------------------------------

        print("\nPredicted Gaussian distribution:")

        counts = Counter(predicted_labels)

        for k,v in counts.items():
            print(
                f"{k}: {v}"
            )

        per_query_results = []

        for query in categories:

            query_ious = []
            image_results = []

            best_iou = -1.0
            best_image = None
            best_gt_mask = None
            best_prediction_mask = None
            best_image_name = None

            print("\n--------------------------------")
            print(f"Evaluating: {query}")

            # Find Gaussians belonging to this object
            selected_indices = [
                i
                for i, label in enumerate(predicted_labels)
                if label == query
            ]

            selected_indices = torch.tensor(
                selected_indices,
                dtype=torch.long
            )

            if len(selected_indices) == 0:
                print("No Gaussians selected. Skipping.")
                continue

            selected_xyz = gaussian_xyz[selected_indices]
            selected_scales = gaussian_scales[selected_indices]

            print(
                f"Selected xyz shape: {selected_xyz.shape}"
            )

            for image_name, masks in image_masks.items():

                if query not in masks:
                    continue

                print(
                    f"  Image: {image_name}"
                )

                rgb_path = (
                    Path(args.dataset_root)
                    / scene
                    / "images"
                    / f"{image_name}.jpg"
                )

                image = load_camera_image(rgb_path)

                gt_mask = cv2.imread(
                    str(masks[query]),
                    cv2.IMREAD_GRAYSCALE
                )

                gt_mask = (
                    gt_mask > 0
                ).astype(np.uint8)

                camera = None

                for img in images.values():
                    if img.name == f"{image_name}.jpg":
                        camera = cameras[img.camera_id]
                        break


                assert gt_mask.shape == (
                    camera.height,
                    camera.width
                ), \
                f"Mask {gt_mask.shape} != Camera {(camera.height,camera.width)}"


                # ------------------------------------------
                # Project selected Gaussians into image
                # ------------------------------------------

                


                projected_pixels, valid_mask, depths = project_gaussians(
                    selected_xyz,
                    images,
                    cameras,
                    f"{image_name}.jpg"
                )

                

                prediction_mask = render_prediction_mask(
                    projected_pixels,
                    depths,
                    valid_mask,
                    torch.arange(len(selected_xyz)),
                    selected_scales.numpy(),
                    gt_mask.shape
                )


                save_mask_visualization(

                    image=image,

                    gt_mask=gt_mask,

                    prediction_mask=prediction_mask,

                    output_dir=scene_output,

                    query=query,

                    image_name=image_name,

                    save_best=False

                )

                

                print(
                    f"Projected Gaussians: {valid_mask.sum()}"
                )

                print(
                    f"Prediction mask pixels: {prediction_mask.sum()}"
                )

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

                scene_results.append(
                    {
                        "object": query,
                        "image": image_name,
                        "iou": iou
                    }
                )

                object_results[query].append(iou)

                print(
                    f"IoU ({query}, {image_name}): {iou:.4f}"
                )

            if len(query_ious) > 0:

                mean_iou = np.mean(query_ious)

            else:

                mean_iou = 0.0

            print(f"Mean IoU ({query}) : {mean_iou:.4f}")

            query_metrics[query] = float(mean_iou)
              
            # ---------------------------------------------------------
            # Save ONLY the best visualization for this query
            # ---------------------------------------------------------

            if best_image is not None:

                save_mask_visualization(
                    image=best_image,
                    gt_mask=best_gt_mask,
                    prediction_mask=best_prediction_mask,
                    output_dir=scene_output,
                    query=query,
                    image_name=best_image_name,
                    save_best=True,
                )

            save_imagewise_results(
                scene_output,
                query,
                image_results
            )

                

            per_query_results.append({

                "method": "SceneSplat",

                "scene": scene,

                "query": query,

                "mean_iou": float(mean_iou),

                "threshold": 0,

                "selected_gaussians": len(selected_indices)

                })

        save_per_query_csv(
            scene_output,
            per_query_results
        )        



        if len(scene_results) > 0:

            mean_iou = np.mean(
                [
                    x["iou"]
                    for x in scene_results
                ]
            )

            print(
                f"\nScene {scene} Mean IoU: {mean_iou:.4f}"
            )

        print("\nMean IoU per object:")

        for obj, values in object_results.items():
            print(
                f"{obj:<20}: {np.mean(values):.4f}"
            )

        features = features.cpu()
        text_embeddings = text_embeddings.cpu()
        similarity = similarity.cpu()

        # Free GPU memory before next scene
        del features
        del text_embeddings
        del similarity

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = compute_global_metrics(
            query_metrics
        )
        
        global_metrics = {
            "scene": scene,
            "method": "SceneSplat",
            "per_query": query_metrics
        }
        
        global_metrics.update(metrics)
        
        save_global_metrics(
            scene_output,
            global_metrics
        )



if __name__ == "__main__":
    main()