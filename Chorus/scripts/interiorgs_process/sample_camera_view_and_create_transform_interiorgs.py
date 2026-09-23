import math
import os
import time
from typing import Literal
import torch

from gsplat import rasterization
import numpy as np
import matplotlib
from tqdm import tqdm
import json
from pathlib import Path
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TKAgg')
from shapely.geometry import Polygon
from shapely.affinity import scale

from utils import (
    load_npy_to_gs,
    camera_pose_to_visibility_mask_interiorgs
)

import argparse

import torch.nn as nn
import sklearn.decomposition
import sklearn
from PIL import Image
from plyfile import PlyData, PlyElement
from typing import Dict
import shutil
from matplotlib.path import Path


CHUNK_SIZE = 256
SH_C0 = 0.28209479177387814
UINT11_MASK = (1 << 11) - 1
UINT10_MASK = (1 << 10) - 1
UINT8_MASK = (1 << 8) - 1

# Original list of labels from 0 to 71
LABEL_LIST = [
    "books", "packaged food", "wall", "produce", "ceiling light", "ceiling", "medicine", "floor", 
    "bedding", "tableware", "decor", "container", "chair", "tobacco", "beverage", "window", 
    "window accessory", "utensils", "wardrobe", "wall art", "plant", "cabinet", "wine", "table", 
    "toiletries", "bakery", "snacks and candy", "door", "towel", "stationery", "vase", "paper goods", 
    "shower", "floor covering", "lamp", "bed", "stool", "mirror", "toy", "ventilation and heating", 
    "faucet", "toilet", "television", "sofa", "column", "shelving", "computer", "refrigerator", 
    "computer peripheral", "mattress", "clock", "desk", "cookware", "trash can", "fan", "monitor", 
    "dishwasher", "oven", "dresser", "bookshelf", "microwave", "washing machine", "countertop", 
    "bathtub", "electronics accessory", "range hood", "cooktop", "drawer", "bench", "stove", 
    "sink", "staircase"
]

# --- Height Definitions ---
# BOTTOM: Z < 0.1m (Structure/covering)
# LOWER: 0.1m < Z < 1.5m (Sitting level, counter level, low storage)
# HIGH: 1.5m < Z < 2.5m (Above eye level, upper cabinets, high windows)
# TOP: Z > 2.5m (Ceiling, fixtures attached to the ceiling)
# ALL: Spans a large vertical extent (e.g., wall, column, tall furniture)

HEIGHT_CATEGORIES = {
    "bottom": [],  # Floor-level only
    "lower": [],   # Low to Mid-height (0 to ~1.5m)
    "high": [],    # Mid to High-height (~1.5m to ~2.5m)
    "top": [],     # Ceiling/Topmost fixtures
    "all": []      # Spans major vertical height
}

# --- Mapping Logic ---
ITEM_MAPPING = {
    # BOTTOM: Floor plane or covering that rests on the floor
    "floor": "bottom",
    "floor covering": "bottom",

    # TOP: Ceiling and fixtures directly attached to the ceiling
    "ceiling light": "top",
    "ceiling": "top",
    
    # ALL: Spans vertical height (Floor to Ceiling or significant wall portion)
    "wall": "all",
    "wardrobe": "all",          # Tall storage
    "door": "all",
    "column": "all",
    "refrigerator": "all",      # Tall appliance
    "staircase": "all",
    "ventilation and heating": "all", # Vents can span wall or floor to ceiling
    "shelving": "all",          # Often spans vertically from low to high
    
    # HIGH: Typically mounted on walls (mid-to-high) or starts high
    "window": "high",
    "window accessory": "high",
    "wall art": "high",
    "mirror": "high",
    "television": "high",
    "clock": "high",
    "range hood": "high",
    
    # LOWER: Everything else (rests on floor/counter/table, under 1.5m or sitting height)
    # Includes furniture, small items, appliances, plumbing
    "books": "lower",
    "packaged food": "lower",
    "produce": "lower",
    "medicine": "lower",
    "bedding": "lower",
    "tableware": "lower",
    "decor": "lower",
    "container": "lower",
    "chair": "lower",
    "tobacco": "lower",
    "beverage": "lower",
    "utensils": "lower",
    "plant": "lower",
    "cabinet": "lower",
    "wine": "lower",
    "table": "lower",
    "toiletries": "lower",
    "bakery": "lower",
    "snacks and candy": "lower",
    "towel": "lower",
    "stationery": "lower",
    "vase": "lower",
    "paper goods": "lower",
    "shower": "lower",
    "lamp": "lower",
    "bed": "lower",
    "stool": "lower",
    "toy": "lower",
    "faucet": "lower",
    "toilet": "lower",
    "sofa": "lower",
    "computer": "lower",
    "computer peripheral": "lower",
    "mattress": "lower",
    "desk": "lower",
    "cookware": "lower",
    "trash can": "lower",
    "fan": "lower",
    "monitor": "lower",
    "dishwasher": "lower",
    "oven": "lower",
    "dresser": "lower",
    "bookshelf": "lower",
    "microwave": "lower",
    "washing machine": "lower",
    "countertop": "lower",
    "bathtub": "lower",
    "electronics accessory": "lower",
    "cooktop": "lower",
    "drawer": "lower",
    "bench": "lower",
    "stove": "lower",
    "sink": "lower",
}

# --- Build the final dictionary mapping ID (0-71) to Category ---
ID_TO_HEIGHT_CATEGORY = {}
for index, item in enumerate(LABEL_LIST):
    # Retrieve the category based on the item name; default to 'lower' if not found.
    category = ITEM_MAPPING.get(item, "lower")  
    ID_TO_HEIGHT_CATEGORY[index] = category

print("--- ID to Height Category Mapping ---")
# The output now prints the desired dictionary mapping 0-71 to its height category string
print(ID_TO_HEIGHT_CATEGORY)

HEIGHT_CATEGORIES["bottom"] = [idx for idx, cat in ID_TO_HEIGHT_CATEGORY.items() if cat == "bottom"]
HEIGHT_CATEGORIES["lower"] = [idx for idx, cat in ID_TO_HEIGHT_CATEGORY.items() if cat == "lower"]
HEIGHT_CATEGORIES["high"] = [idx for idx, cat in ID_TO_HEIGHT_CATEGORY.items() if cat == "high"]
HEIGHT_CATEGORIES["top"] = [idx for idx, cat in ID_TO_HEIGHT_CATEGORY.items() if cat == "top"]
HEIGHT_CATEGORIES["all"] = [idx for idx, cat in ID_TO_HEIGHT_CATEGORY.items() if cat == "all"]



# raise NotImplementedError("This script is for generating the ID to Height Category mapping only.")


def _detach_tensors_from_dict(d, inplace=True):
    if not inplace:
        d = d.copy()
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].detach()
    return d

def farthest_point_sampling(points: np.ndarray, K: int, MIN_DISTANCE_THRESHOLD=1.0) -> np.ndarray:
    """
    Performs Farthest Point Sampling on a set of 2D points with a minimum distance constraint.

    Args:
        points (np.ndarray): The input point cloud, shape (N, 2).
        K (int): The maximum number of points to sample (K <= N).

    Returns:
        np.ndarray: The K sampled points, shape (K', 2), where K' <= K.
    """
    N = points.shape[0]
    
    # --- New Constraint Threshold ---
    MIN_DIST_SQ = MIN_DISTANCE_THRESHOLD ** 2 # Use squared distance for optimization
    
    if K >= N:
        print("Warning: K is greater than or equal to N. Returning all points.")
        return points

    # --- 1. Initialization ---
    
    # Stores the indices of the sampled points. We start with K max size.
    sample_indices_list = []
    
    # Stores the squared distance from each point to the nearest sampled point.
    distances = np.full(N, np.inf)

    # 2. Randomly select the first point (index 0)
    farthest_idx = np.random.randint(N)
    sample_indices_list.append(farthest_idx)
    
    print(f"Starting FPS with max K={K} samples from N={N} points.")
    print(f"Minimum distance required between samples: {MIN_DISTANCE_THRESHOLD}m")
    
    # --- 3. Iterative Selection ---
    
    # Start loop from the second point (k=1) up to K
    for k in range(1, K):
        # The index of the last point added to the set
        current_sample_idx = sample_indices_list[-1]
        
        # Calculate the distance from ALL points to the NEWEST sampled point.
        last_point = points[current_sample_idx]
        
        # Calculate squared Euclidean distance: ||P_i - P_last||^2
        new_distances = np.sum((points - last_point) ** 2, axis=1)
        
        # Update the minimum distance array (distances[i] = distance to nearest neighbor in sample_indices)
        distances = np.minimum(distances, new_distances)
        
        # Find the point that has the maximum minimum distance (the farthest point)
        farthest_idx = np.argmax(distances)
        max_min_distance_sq = distances[farthest_idx]
        
        # --- THRESHOLD CHECK (NEW) ---
        if max_min_distance_sq < MIN_DIST_SQ:
            print(f"\nSTOPPED at k={k}: Maximum separation achieved ({np.sqrt(max_min_distance_sq):.4f}m) is less than the minimum required distance ({MIN_DISTANCE_THRESHOLD}m).")
            break
        
        # Add the farthest point's index to the sample set
        sample_indices_list.append(farthest_idx)

    # --- 4. Return the sampled points ---
    sample_indices = np.array(sample_indices_list)
    sampled_points = points[sample_indices]
    
    print(f"\nSampling complete. Total points sampled (K'): {len(sampled_points)}")
    return sampled_points


def _read_supersplat_gaussian_ply(ply: PlyData) -> Dict[str, np.ndarray]:
    chunk = ply["chunk"].data
    vertex = ply["vertex"].data
    num = vertex.shape[0]

    chunk_indices = np.arange(num, dtype=np.int64) // CHUNK_SIZE
    chunk_indices = np.minimum(chunk_indices, len(chunk) - 1)

    min_x = chunk["min_x"][chunk_indices]
    min_y = chunk["min_y"][chunk_indices]
    min_z = chunk["min_z"][chunk_indices]
    max_x = chunk["max_x"][chunk_indices]
    max_y = chunk["max_y"][chunk_indices]
    max_z = chunk["max_z"][chunk_indices]

    min_scale_x = chunk["min_scale_x"][chunk_indices]
    min_scale_y = chunk["min_scale_y"][chunk_indices]
    min_scale_z = chunk["min_scale_z"][chunk_indices]
    max_scale_x = chunk["max_scale_x"][chunk_indices]
    max_scale_y = chunk["max_scale_y"][chunk_indices]
    max_scale_z = chunk["max_scale_z"][chunk_indices]

    min_r = chunk["min_r"][chunk_indices]
    min_g = chunk["min_g"][chunk_indices]
    min_b = chunk["min_b"][chunk_indices]
    max_r = chunk["max_r"][chunk_indices]
    max_g = chunk["max_g"][chunk_indices]
    max_b = chunk["max_b"][chunk_indices]

    packed_position = vertex["packed_position"].astype(np.uint32)
    packed_scale = vertex["packed_scale"].astype(np.uint32)
    packed_rotation = vertex["packed_rotation"].astype(np.uint32)
    packed_color = vertex["packed_color"].astype(np.uint32)

    px = (packed_position >> 21) & UINT11_MASK
    py = (packed_position >> 11) & UINT10_MASK
    pz = packed_position & UINT11_MASK
    px = px.astype(np.float32) / UINT11_MASK
    py = py.astype(np.float32) / UINT10_MASK
    pz = pz.astype(np.float32) / UINT11_MASK

    coord = np.empty((num, 3), dtype=np.float32)
    coord[:, 0] = min_x * (1.0 - px) + max_x * px
    coord[:, 1] = min_y * (1.0 - py) + max_y * py
    coord[:, 2] = min_z * (1.0 - pz) + max_z * pz

    sx = (packed_scale >> 21) & UINT11_MASK
    sy = (packed_scale >> 11) & UINT10_MASK
    sz = packed_scale & UINT11_MASK
    sx = sx.astype(np.float32) / UINT11_MASK
    sy = sy.astype(np.float32) / UINT10_MASK
    sz = sz.astype(np.float32) / UINT11_MASK

    scale_log = np.empty((num, 3), dtype=np.float32)
    scale_log[:, 0] = min_scale_x * (1.0 - sx) + max_scale_x * sx
    scale_log[:, 1] = min_scale_y * (1.0 - sy) + max_scale_y * sy
    scale_log[:, 2] = min_scale_z * (1.0 - sz) + max_scale_z * sz
    scale = scale_log
    # np.exp(scale_log)

    norm = np.float32(1.0 / (np.sqrt(2.0) * 0.5))
    a = ((packed_rotation >> 20) & UINT10_MASK).astype(np.float32) / UINT10_MASK
    b = ((packed_rotation >> 10) & UINT10_MASK).astype(np.float32) / UINT10_MASK
    c = (packed_rotation & UINT10_MASK).astype(np.float32) / UINT10_MASK
    a = (a - 0.5) * norm
    b = (b - 0.5) * norm
    c = (c - 0.5) * norm
    m = np.sqrt(np.maximum(0.0, 1.0 - (a * a + b * b + c * c)))
    which = packed_rotation >> 30

    quat = np.empty((num, 4), dtype=np.float32)
    mask = which == 0
    if np.any(mask):
        quat[mask, 0] = m[mask]
        quat[mask, 1] = a[mask]
        quat[mask, 2] = b[mask]
        quat[mask, 3] = c[mask]
    mask = which == 1
    if np.any(mask):
        quat[mask, 0] = a[mask]
        quat[mask, 1] = m[mask]
        quat[mask, 2] = b[mask]
        quat[mask, 3] = c[mask]
    mask = which == 2
    if np.any(mask):
        quat[mask, 0] = a[mask]
        quat[mask, 1] = b[mask]
        quat[mask, 2] = m[mask]
        quat[mask, 3] = c[mask]
    mask = which == 3
    if np.any(mask):
        quat[mask, 0] = a[mask]
        quat[mask, 1] = b[mask]
        quat[mask, 2] = c[mask]
        quat[mask, 3] = m[mask]

    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-9
    quat = quat / quat_norm
    sign = np.sign(quat[:, 0])
    quat = quat * sign[:, None]

    cr = ((packed_color >> 24) & UINT8_MASK).astype(np.float32) / UINT8_MASK
    cg = ((packed_color >> 16) & UINT8_MASK).astype(np.float32) / UINT8_MASK
    cb = ((packed_color >> 8) & UINT8_MASK).astype(np.float32) / UINT8_MASK
    cw = (packed_color & UINT8_MASK).astype(np.float32) / UINT8_MASK

    r = min_r * (1.0 - cr) + max_r * cr
    g = min_g * (1.0 - cg) + max_g * cg
    b = min_b * (1.0 - cb) + max_b * cb
    fdc = np.stack([(r - 0.5) / SH_C0, (g - 0.5) / SH_C0, (b - 0.5) / SH_C0], axis=-1)
    color = np.clip(fdc * SH_C0 + 0.5, 0.0, 1.0) * 255.0
    color = color.astype(np.uint8)

    opacity = np.clip(cw, 0.0, 1.0).astype(np.float32)


    features_dc = (color / 255.0 - 0.5) / 0.28209479177387814  # map to [-1.77, 1.77]
    features_dc = features_dc.reshape(-1, 3, 1)
    scales = scale.reshape(-1, 3)
    rots = quat.reshape(-1, 4)
    opacity = opacity.reshape(-1, 1)
    # features_extra = np.zeros((xyz.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))# sh degree 0, extra is 0 dimension
    max_sh_degree = 0
    splats = {
        "active_sh_degree": max_sh_degree,
        "means": torch.tensor(coord).float().cuda(),
        "features_dc": torch.tensor(features_dc).float().cuda().transpose(1, 2).contiguous(),
        # "features_rest": torch.tensor(features_extra).float().cuda().transpose(1, 2).contiguous(),
        "scaling": torch.tensor(scales).float().cuda(),
        "rotation": torch.tensor(rots).float().cuda(),
        "opacity": torch.tensor(opacity).float().cuda().squeeze(1),
    }


    _detach_tensors_from_dict(splats)


    return splats

def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature



def main(
    interior_gs_path: str = "./interior_gs/scenes",
    interior_gs_preprocessed_path: str = "./interior_gs_preprocessed",
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or GSplat for checkpoints
    sampling_positions=20,
    fx = 500.0,
    fy = 500.0,
    H = 480,
    W = 640,
    filter_by_depth=True
):

    splits = ["train", "val", "test"]

    for split_i in splits:
        data_root_path = os.path.join(interior_gs_preprocessed_path, split_i)
        if not os.path.exists(data_root_path):
            print("Not found ", data_root_path)
            continue
        scenes = sorted(os.listdir(data_root_path))
        # check if it is folder 
        # ignore fold .cache or hidden folder
        scenes = [s for s in scenes if os.path.isdir(os.path.join(data_root_path, s))]
        scenes = [s for s in scenes if not s.startswith('.')]
        scenes = sorted(scenes)

        structure = np.ones((3,3))
        eight_direction_rotations = []
        maximal_allow_to_occupied_distance = 1.5


        for scene_i in scenes:

            gs_npy_path_i = os.path.join(data_root_path, scene_i)
            splats = load_npy_to_gs(
                gs_npy_path_i, rasterizer=rasterizer, dataset="interiorgs"
            )

            segment = np.load(os.path.join(gs_npy_path_i, 'segment.npy'))

            ### downsample the splts xyz for faster process, first we sample xy location (ignore height)
            xyz = splats['means'].cpu().numpy()
            xyz_downsample_indices = np.random.choice(xyz.shape[0], size=xyz.shape[0]//20, replace=False)
            xyz_downsample = xyz[xyz_downsample_indices, :]
            segment_downsample = segment[xyz_downsample_indices]
            print(" x range", xyz[:,0].min(), xyz[:,0].max())
            print(" y range", xyz[:,1].min(), xyz[:,1].max())
            print(" z range", xyz[:,2].min(), xyz[:,2].max())
            z_range = xyz[:,2].max() - xyz[:,2].min()


            xy_downsample = xyz_downsample[:, :2]
            try:
                hull=ConvexHull(xy_downsample)
            except Exception as e:
                print("Convex hull error:", e)
                continue
            hull_vertices = xy_downsample[hull.vertices, :]
            original_polygon = Polygon(hull_vertices)
            EROSION_DISTANCE = -1.25  # in meters, negative for erosion
            eroded_polygon = original_polygon.buffer(EROSION_DISTANCE)
            # Check if the polygon collapsed (too much erosion)
            if eroded_polygon.is_empty:
                print(f"\nWarning: Erosion of {abs(EROSION_DISTANCE)}m caused the polygon to collapse. Using original hull for query.")
                final_query_polygon = original_polygon
            else:
                print(f"\nHull eroded inward by {abs(EROSION_DISTANCE)} meters.")
                final_query_polygon = eroded_polygon

            if final_query_polygon.geom_type == 'Polygon':
                final_hull_vertices = np.array(final_query_polygon.exterior.coords)
            elif final_query_polygon.geom_type == 'MultiPolygon':
                # If multipolygon, just take the largest component or simplify.
                # For simplicity, we'll take the exterior of the first component.
                final_hull_vertices = np.array(final_query_polygon.geoms[0].exterior.coords)
            else:
                final_hull_vertices = hull_vertices # Fallback

            polygon_path = Path(final_hull_vertices)
            xy_hull_area = final_query_polygon.area
            # xy_hull_area = hull.volume
            offset_sampling_positions = int(xy_hull_area / 25.0)  # every 5mx5m area, we add one more sampling position
            sampling_positions_adjust = offset_sampling_positions + sampling_positions
            print("Convex hull area: ", xy_hull_area, " m^2", " Additional sampling positions due to area: ", offset_sampling_positions, " total sampling positions: ", sampling_positions_adjust)

            gs_raw_path_i = os.path.join(interior_gs_path, scene_i)
            occupancy_png_path = os.path.join(gs_raw_path_i, 'occupancy.png')
            occupancy_meta_path = os.path.join(gs_raw_path_i, 'occupancy.json')

            occupancy_meta = json.load(open(occupancy_meta_path, 'r'))
            occupancy_img = Image.open(occupancy_png_path)
            occupancy_img_np = np.array(occupancy_img) 
            occupancy_free_mask = np.zeros_like(occupancy_img_np)
            occupancy_free_mask[occupancy_img_np==255] = 1

            occupancy_W = occupancy_free_mask.shape[1]
            occupancy_H = occupancy_free_mask.shape[0]
            x_range = occupancy_meta['upper'][0] - occupancy_meta['lower'][0]
            y_range = occupancy_meta['upper'][1] - occupancy_meta['lower'][1]
            x_scale = occupancy_meta['scale']  #x_range / occupancy_W
            y_scale = occupancy_meta['scale'] #y_range / occupancy_H

            # we shirnk the occupancy map a bit to avoid boundary issues
            iteration = 10
            occupancy_free_mask_shirnk = binary_erosion(occupancy_free_mask, structure=structure, iterations=iteration).astype(occupancy_free_mask.dtype)
            occupancy_free_mask_shirnk[:int(0.1*occupancy_free_mask_shirnk.shape[0]), :] = 0
            occupancy_free_mask_shirnk[-int(0.1*occupancy_free_mask_shirnk.shape[0]):, :] = 0
            occupancy_free_mask_shirnk[:, :int(0.1*occupancy_free_mask_shirnk.shape[1])] = 0
            occupancy_free_mask_shirnk[:, -int(0.1*occupancy_free_mask_shirnk.shape[1]):] = 0

            # 10 pixel * 0.05 = 0.5 meter

            # save occupancy_free_mask_shirnk for debug
            occupancy_free_mask_shirnk_img = Image.fromarray((occupancy_free_mask_shirnk * 255).astype(np.uint8))
            occupancy_free_mask_shirnk_img.save(os.path.join(data_root_path, scene_i, 'occupancy_free_mask_shirnk.png'))

            # now we start sampling points in free space given mask 
            free_y, free_x = np.where(occupancy_free_mask_shirnk==1)
            # I want to sample points with fps so it spread out
            fps_sampled_points = farthest_point_sampling(np.stack([free_x, free_y], axis=1), sampling_positions_adjust)
            sampled_x = fps_sampled_points[:,0]
            sampled_y = fps_sampled_points[:,1]
            
            # we check if sampled x and y is valid 
            valid_camera_positions = []
            
            for i, (sx, sy) in enumerate(zip(sampled_x, sampled_y)):
                ray_direction_vectors = np.array([0,1]).reshape(2,1)  # initial ray direction is along y axis (forward)

                # check if sx sy in the hull
                tx = (-(sx) * x_scale + occupancy_meta['upper'][0])
                ty = ((sy) * y_scale + occupancy_meta['lower'][1])
                if not polygon_path.contains_point((tx, ty)):
                    print(f"Sampled camera position {i} at pixel ({sx}, {sy}) is outside the convex hull, discard this position.")
                    continue


                valid_camera_positions_temp = []
                valid_camera_positions_temp_dist = []
                for angle in range(0, 360, 45):
                    theta = math.radians(angle)
                    # we assume now in 2D, the x axis is right, and future x, y axis is forward, and future z, z axis is up, and future -y
                    R_2D = np.array([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta),     math.cos(theta)]])
                    
                    ray_direction_vectors_rotated = R_2D @ ray_direction_vectors  # shape (2,1)
                    # we check in this direction, how close is the nearest occupied space
                    check_distance = 0
                    maximal_steps = 40 #maximal_allow_to_occupied_distance / occupancy_meta['scale']  # 1 meter, to pixel range
                    while check_distance < maximal_steps:
                        check_x = int(sx + ray_direction_vectors_rotated[0,0] * check_distance)
                        check_y = int(sy + ray_direction_vectors_rotated[1,0] * check_distance)
                        if check_x <0 or check_x >= occupancy_free_mask.shape[1] or check_y <0 or check_y >= occupancy_free_mask.shape[0]:
                            # out of bound, consider as occupied
                            break
                        if occupancy_free_mask[check_y, check_x] == 0:
                            # hit occupied space
                            # print("hit occupied space at distance ", check_distance * occupancy_meta['scale'])
                            break
                        check_distance +=1

                    if check_distance * occupancy_meta['scale'] < maximal_allow_to_occupied_distance:
                        print(f"Sampled camera position {i} at pixel ({sx}, {sy}) is too close to occupied space in angle {angle}, discard this position.")
                    else:
                        valid_camera_positions_temp.append((sx, sy, angle))
                        valid_camera_positions_temp_dist.append(check_distance * occupancy_meta['scale'])
                        print("Sampled camera position ", i, " at pixel (", sx, ",", sy, ") is valid with angle ", angle)

                # else:
                for item in valid_camera_positions_temp:
                    valid_camera_positions.append(item)

                

            from PIL import ImageDraw
            occupancy_img_with_samples = occupancy_img.convert("RGB")
            draw = ImageDraw.Draw(occupancy_img_with_samples)
            # for sx, sy in zip(sampled_x, sampled_y):
            #     draw.ellipse((sx-3, sy-3, sx+3, sy+3), fill=(255,0,0))
            for (sx, sy, angle) in valid_camera_positions:
                draw.ellipse((sx-3, sy-3, sx+3, sy+3), fill=(255,0,0))
                # draw angle
                length = 10
                theta = math.radians(angle)
                end_x = sx - length * math.sin(theta)
                end_y = sy + length * math.cos(theta)
                draw.line((sx, sy, end_x, end_y), fill=(0,255,0), width=2)

            occupancy_img_with_samples.save(os.path.join(data_root_path, scene_i, 'occupancy_with_sampled_camera_positions_filter_angle.png'))
            plt.close('all')

            # raise NotImplementedError("Debugging convex hull visualization done.")

            # now we rewrite to to camera transformers.json file 
            transformers = {"fl_x": fx,
                            "fl_y": fy,
                            "cx": W / 2.0,
                            "cy": H / 2.0,
                            "w": W,
                            "h": H,
                            "frames": []}
            steps = 0
            for (sx, sy, angle) in valid_camera_positions:
                theta = math.radians(angle)
                R_c2w_2D = np.array([[math.cos(theta), -math.sin(theta),0],
                                [math.sin(theta),     math.cos(theta),0],
                                [0,                   0,               1]]) 

                # now we swap axis, z to y, y to -z
                swap_matrix = np.array([[1,0,0],
                                        [0,0,1],
                                        [0,-1,0]])
                R_c2w =  R_c2w_2D @ swap_matrix

                tx = (-(sx) * x_scale + occupancy_meta['upper'][0])
                ty = ((sy) * y_scale + occupancy_meta['lower'][1])
                # tz = z_middle + 0.5 # favor higher camera height  

                gs_xy =xyz_downsample[:, :2]
                # xyz
                dists = np.linalg.norm(gs_xy - np.array([tx, ty]).reshape(1,2), axis=1)
                within_indices = np.where(dists < 4.0)[0]

                t0 = time.time()
                # we find splats close to this  camera pose in xy coordinate 
                if len(within_indices) ==0:
                    # this means no splat within 4 meter, this is very open, we ignore this camera pose 
                    print(f"Scene {scene_i} camera position at pixel ({sx}, {sy}) has no nearby splats within 4 meter, ignore this position.")
                    continue

                else:
                    within_indices_z = xyz_downsample[within_indices, 2]
                    within_segment = segment_downsample[within_indices]
                    # now we check if neary by exist bottom, wall, ceiling splats
                    # ignore -1 which is not labelled
                    within_indices_z = within_indices_z[within_segment>=0]
                    within_segment = within_segment[within_segment>=0]
                    
                    
                    within_segment_unique = np.unique(within_segment)
                    # we start from bottom splats
                    # tz_candidate_bottom = within_indices_z.min()
                    tz_candidate_bottom = None
                    for bottom_id in HEIGHT_CATEGORIES['bottom']:
                        if bottom_id in within_segment_unique:
                            # we find closeby bottm splats, we add 2 meter of this points maximal to be z
                            tz_candidate_bottom = within_indices_z[within_segment==bottom_id].max() + 1.5

                   #  tz_candidate_lower =  within_indices_z.min() + 2.0
                    tz_candidate_lower = None
                    for lower_id in HEIGHT_CATEGORIES['lower']:
                        if lower_id in within_segment_unique:
                            # we find closeby lower splats, we add 1 meter of this points maximal to be z
                            tz_candidate_lower = within_indices_z[within_segment==lower_id].min() + 0.75

                    # tz_candidate_high =  within_indices_z.max() -2.0
                    tz_candidate_high = None
                    for high_id in HEIGHT_CATEGORIES['high']:
                        if high_id in within_segment_unique:
                            # we find closeby high splats, we minus 1 meter of this points minimal to be z
                            tz_candidate_high = within_indices_z[within_segment==high_id].min() + 0.75

                    # tz_candidate_top =  within_indices_z.max() -2.0
                    tz_candidate_top = None
                    for top_id in HEIGHT_CATEGORIES['top']:
                        if top_id in within_segment_unique:
                            # we find closeby top splats, we minus 2 meter of this points minimal to be z
                            tz_candidate_top = within_indices_z[within_segment==top_id].min() - 2.0
                        
                    # t_all_candidates = np.mean(within_indices_z.min() + within_indices_z.max()) /2.0
                    # t_all_candidates = None
                    # for all_id in HEIGHT_CATEGORIES['all']:
                    #     if all_id in within_segment_unique:
                    #         t_all_candidates = within_indices_z[within_segment==all_id].mean()
                    
                    # now we try to find one best from all candidates if not none
                    # first we not sure about the ceil if it is moving
                    tz = None
                    t_select = 0
                    if tz_candidate_top is not None:
                        tz = tz_candidate_top
                        t_select = 1
                    # then we consider high and lower
                    if tz_candidate_lower is not None:
                        tz = tz_candidate_lower
                        t_select =2

                    if tz_candidate_high is not None:
                        tz = tz_candidate_high
                        t_select = 3
                    # we being more conservative with bottom
                    if tz_candidate_bottom is not None:
                        tz = tz_candidate_bottom
                        t_select = 4

                    if tz is None:
                        # tz still None, now we manual set height to be mean height
                        tz = (xyz_downsample[:,2].min() + xyz_downsample[:,2].max()) /2.0
                        # print("Using mean height for camera position at pixel (", sx, ",", sy, ")")
                        t_select = 5
                
                if t_select ==1:
                    print(f"height decided to be TOP {tz:.2f} for camera position at pixel ({sx}, {sy}) time used {time.time() - t0:.4f}s")
                elif t_select ==2:
                    print(f"height decided to be LOWER {tz:.2f} for camera position at pixel ({sx}, {sy}) time used {time.time() - t0:.4f}s")
                elif t_select ==3:
                    print(f"height decided to be HIGH {tz:.2f} for camera position at pixel ({sx}, {sy}) time used {time.time() - t0:.4f}s")
                elif t_select ==4:
                    print(f"height decided to be BOTTOM {tz:.2f} for camera position at pixel ({sx}, {sy}) time used {time.time() - t0:.4f}s")
                else:
                    print(f"height decided to be MEAN {tz:.2f} for camera position at pixel ({sx}, {sy}) time used {time.time() - t0:.4f}s")
                # print("height decided to be ", tz, " for camera position at pixel (", sx, ",", sy, ")", " time used ", time.time() - t0)      
                

                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = R_c2w
                transform_matrix[:3, 3] = np.array([tx, ty, tz])
                transformers["frames"].append({
                    "file_path": f"frame{steps:06d}.png",
                    "transform_matrix": transform_matrix.tolist()
                })
                steps +=1

            # save transformers
            transformers_save_path = os.path.join(data_root_path, scene_i, 'transforms_camera_positions.json')
            with open(transformers_save_path, 'w') as f:
                json.dump(transformers, f, indent=4)

            # remove old images if exist
            shutil.rmtree(os.path.join(gs_npy_path_i, 'render'), ignore_errors=True)
            os.makedirs(os.path.join(gs_npy_path_i, 'render',), exist_ok=True)
            camera_pose_to_visibility_mask_interiorgs(splats, gs_npy_path_i)

    
    #### doing filter out based on depth
    
    if filter_by_depth:
        meta_data = {}
        for split_i in splits:
            meta_data[split_i] = []
            split_path = os.path.join(interior_gs_preprocessed_path, split_i)
            if not os.path.exists(split_path):
                print("Not found ", split_path)
                continue
            all_scenes = os.listdir(split_path)
            # only save dir and not start with .
            all_scenes = [scene for scene in all_scenes if os.path.isdir(os.path.join(split_path, scene)) and not scene.startswith('.')]
            all_scenes = sorted(all_scenes)
            for scene_i in all_scenes:

                valid_idx = []
                scene_i_path = os.path.join(split_path, scene_i)
                print("filtering scene:", split_i, scene_i)
                transforms_camera_positions_path = os.path.join(scene_i_path, 'transforms_camera_positions.json')
                render_images_root_path = os.path.join(scene_i_path, 'render')
                visiable_gaussian_masks = os.path.join(scene_i_path,  'visiable_gaussian_masks_per_frame.npy')

                transforms_camera_positions = json.load(open(transforms_camera_positions_path, 'r'))
                visiable_gaussian_masks = np.load(visiable_gaussian_masks, allow_pickle=True)

                valid_frames = []
                for frame_idx in range(len(transforms_camera_positions['frames'])):
                    visiable_gaussian_mask = visiable_gaussian_masks[frame_idx]
                    # calculate the ratio of visiable gaussian
                    visiable_ratio = np.sum(visiable_gaussian_mask) / visiable_gaussian_mask.size
                    frames_name = transforms_camera_positions['frames'][frame_idx]['file_path'].split('.png')[0]
                    
                    depth_image_path = os.path.join(render_images_root_path, frames_name + '_depth.png')
                    # open depth image
                    depth_image = np.array(Image.open(depth_image_path))
                    
                    depth_image_meter = depth_image / 10.

                    depth_image_meter_center = depth_image_meter[depth_image_meter.shape[0]//4: depth_image_meter.shape[0]*3//4,
                                                                depth_image_meter.shape[1]//4: depth_image_meter.shape[1]*3//4]
                    avg_depth = np.mean(depth_image_meter_center)
                    median_depth = np.median(depth_image_meter_center)

                    close_depth_ratio = (np.sum(depth_image_meter_center < 1.0) ) / depth_image_meter_center.size
                    too_close_ratio = (np.sum(depth_image_meter_center < 0.5) ) / depth_image_meter_center.size
                    min_depth = np.min(depth_image_meter)


                    print("visiable_ratio:", visiable_ratio, frames_name)
                    # print("depth image shape:", depth_image.shape, np.min(depth_image_meter), np.max(depth_image_meter))
                    print("avg_depth:", avg_depth, "median_depth:", median_depth, "min_depth:", min_depth)
                    print("close depth ratio (<0.5m):", close_depth_ratio)
                    print("too close depth ratio (<0.5m):", too_close_ratio)
                    # first, we think lower than 1% gaussian visiable is bad
                    if np.sum(visiable_gaussian_mask) < 204800/4:
                        print("filtering due to too high gaussian ratio frame:", frames_name)
                        continue

                    # if visiable num too large,we also ignore
                    if np.sum(visiable_gaussian_mask) > 204800*8:
                        print("filtering due to large visiable gaussian num frame:", frames_name)
                        continue
                        # continue
                    # then we think if average depth less than 0.5 meter is bad
                    if avg_depth < 0.1:
                        print("filtering due to low average depth:", frames_name)
                        continue

                    # if 0.5 meter close depth ratio is larger than 25%, we think it is bad
                    if close_depth_ratio > 0.05:
                        print("filtering due to large close depth ratio:", frames_name)
                        continue
                    
                    # if too_close_ratio > 0.05:
                    #     print("filtering due to large too close depth ratio:", frames_name)
                    #     continue

                    valid_frames.append(frame_idx)

                # now we create new folder with only valid frames, and new valid transforms_camera_positions.json and npy
                transforms_camera_positions_filtered = {}
                transforms_camera_positions_filtered['frames'] = []
                # except for key "frames", others are the same
                for key in transforms_camera_positions:
                    if key != 'frames':
                        transforms_camera_positions_filtered[key] = transforms_camera_positions[key]
                    else:
                        for valid_frame_idx in valid_frames:
                            transforms_camera_positions_filtered['frames'].append(transforms_camera_positions['frames'][valid_frame_idx])

                # save new json 
                with open(os.path.join(scene_i_path, 'transforms_camera_positions_filtered.json'), 'w') as f:
                    json.dump(transforms_camera_positions_filtered, f, indent=4)
                # save new npy
                visiable_gaussian_masks_filtered = visiable_gaussian_masks[valid_frames]
                np.save(os.path.join(scene_i_path, 'visiable_gaussian_masks_per_frame_filtered.npy'), visiable_gaussian_masks_filtered)

                # symlink the figures and depth images
                render_images_root_path_filtered = os.path.join(scene_i_path, 'render_filtered')
                shutil.rmtree(render_images_root_path_filtered, ignore_errors=True)
                os.makedirs(render_images_root_path_filtered, exist_ok=True)
                for valid_frame_idx in valid_frames:
                    frames_name = transforms_camera_positions['frames'][valid_frame_idx]['file_path'].split('.png')[0]
                    src_image_path = os.path.join(render_images_root_path, frames_name + '.png')
                    src_depth_path = os.path.join(render_images_root_path, frames_name + '_depth.png')
                    dst_image_path = os.path.join(render_images_root_path_filtered, frames_name + '.png')
                    dst_depth_path = os.path.join(render_images_root_path_filtered, frames_name + '_depth.png')
                    shutil.copy2(src_image_path, dst_image_path)
                    shutil.copy2(src_depth_path, dst_depth_path)

                

                print("valid frames num:", len(valid_frames), "out of total:", len(transforms_camera_positions['frames']))
                print("valid frames:", valid_frames)
            
            

def get_arguments():
    argparser = argparse.ArgumentParser(description="Feature Field Extraction")
    argparser.add_argument(
        "--interior_gs_path",
        type=str,
        help="Path to the raw interoirgs file (we need oocupancy mask",
    )
    argparser.add_argument(
        "--interior_gs_preprocessed_path",
        type=str,
        help="Path to the preprocessed interoirgs file (3D format npy)",
    )
    return argparser.parse_args()



if __name__ == "__main__":
    args = get_arguments()
    main(
        interior_gs_path=args.interior_gs_path,
        interior_gs_preprocessed_path=args.interior_gs_preprocessed_path
    )
