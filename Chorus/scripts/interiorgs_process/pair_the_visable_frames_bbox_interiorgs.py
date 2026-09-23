import numpy as np
import os 


import numpy as np
from scipy.spatial import cKDTree
import json
from tqdm import tqdm
import time 
import torch
from tqdm import trange
import torch_cluster
import miniball

import open3d as o3d
import math


def check_points_in_rotated_bbox(query_points: np.ndarray, bbox_params: dict) -> np.ndarray:
    """
    Checks if a set of Mx2 query points is contained within the rotated bounding box.

    Args:
        query_points (np.ndarray): An Mx2 NumPy array of (x, y) coordinates to check.
        bbox_params (dict): The dictionary returned by exhaustive_rotated_bbox_search.

    Returns:
        np.ndarray: A boolean array of shape (M,) where True means the point is inside.
    """
    if query_points.shape[1] != 2:
        raise ValueError("Input 'query_points' array must have shape (M, 2).")

    # 1. Retrieve Bounding Box Parameters
    center_point = bbox_params['center']
    best_angle_deg = bbox_params['best_angle_deg']
    
    # Width and Height are calculated relative to the centered, rotated points.
    width = bbox_params['width']
    height = bbox_params['height']

    # 2. Translate Points (Shift Origin)
    # Move the query points so the box center is at (0, 0).
    points_centered = query_points - center_point
    
    # 3. Define Inverse Rotation Matrix
    # We rotate the *points* by the negative of the box angle to align them 
    # with the box's principal axes.
    angle_rad = math.radians(best_angle_deg) # Negative angle for inverse rotation
    
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    # Rotation matrix for rotating the points back to the AABB space
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # 4. Apply Inverse Rotation
    # (M, 2) @ (2, 2) -> (M, 2)
    points_aligned = points_centered @ rotation_matrix.T
    
    # 5. Perform Axis-Aligned Bounding Box (AABB) Check
    
    # In the rotated space, the box spans from [-width/2, width/2] and [-height/2, height/2].
    half_width = width / 2.0
    half_height = height / 2.0
    
    # Check if the aligned x-coordinates are within [-half_width, half_width]
    is_x_in_range = (points_aligned[:, 0] >= -half_width) & \
                    (points_aligned[:, 0] <= half_width)
    
    # Check if the aligned y-coordinates are within [-half_height, half_height]
    is_y_in_range = (points_aligned[:, 1] >= -half_height) & \
                    (points_aligned[:, 1] <= half_height)
    
    # A point is inside the box only if both conditions are True
    is_inside_mask = is_x_in_range & is_y_in_range
    
    return is_inside_mask


def exhaustive_rotated_bbox_search(points: np.ndarray, center_point: np.ndarray, total_points: np.ndarray):
    """
    Performs an exhaustive search by rotating points every 15 degrees 
    to find the minimum area axis-aligned bounding box (AABB).

    Args:
        points (np.ndarray): An Nx2 NumPy array of (x, y) coordinates.
        center_point (np.ndarray): The 1x2 or 2-element array defining the center 
                                   point used for normalization/unrotation.

    Returns:
        dict: A dictionary containing the best angle (degrees), minimum area, 
              width, and height.
    """
    if points.shape[1] != 2:
        raise ValueError("Input 'points' array must have shape (N, 2).")

    # 1. Normalize: Shift points so the center_point is at the origin (0, 0)
    points_centered = points - center_point
    
    # Initialize best result variables
    min_area = float('inf')
    min_inside_points = 1e8
    best_angle_deg = 0.0
    best_width = 0.0
    best_height = 0.0

    # Iterate through angles from 0 to 165 degrees, stepping by 15 degrees.
    # Checking up to 180 degrees is redundant due to symmetry.
    for angle_deg in np.arange(0, 180, 10):
        angle_deg = float(angle_deg)
        angle_rad = math.radians(angle_deg)
        
        # 2. Define the 2D Rotation Matrix
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        
        # Rotation matrix for rotating the space clockwise (to align the box axes)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # 3. Apply Rotation (Vectorized: [N, 2] @ [2, 2] -> [N, 2])
        # This rotates all points simultaneously
        points_rotated = points_centered @ rotation_matrix.T
        
        # 4. Calculate Axis-Aligned Bounding Box (AABB) in the rotated space
        
        # Find min/max coordinates in the rotated space
        # min_x_rot = np.min(points_rotated[:, 0])
        # max_x_rot = np.max(points_rotated[:, 0])
        # min_y_rot = np.min(points_rotated[:, 1])
        # max_y_rot = np.max(points_rotated[:, 1])
        max_x_rot = np.max(np.abs(points_rotated[:, 0]))
        max_y_rot = np.max(np.abs(points_rotated[:, 1]))
        
        # Calculate Width and Height
        current_width = 2 * max_x_rot
        current_height = 2 * max_y_rot

        # total_within_index = check_points_in_rotated_bbox(total_points, {
        #     "center": center_point,
        #     "best_angle_deg": angle_deg,
        #     "width": current_width,
        #     "height": current_height,
        # })
        current_area = current_width * current_height

        # if np.sum(total_within_index) <= min_inside_points:
        #     min_inside_points = np.sum(total_within_index)
        #     best_angle_deg = angle_deg
        #     best_width = current_width
        #     best_height = current_height

        # current_width = max_x_rot - min_x_rot
        # current_height = max_y_rot - min_y_rot
        # current_area = current_width * current_height
        
        # 5. Check if this is the minimum area found so far
        if current_area < min_area:
            min_area = current_area
            best_angle_deg = angle_deg
            best_width = current_width
            best_height = current_height

    return {
        "best_angle_deg": best_angle_deg,
        "width": best_width,
        "height": best_height,
        "center": center_point,
    }

def get_argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_root_path', type=str, required=True, help='Path to the input PLY file containing the point cloud.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing scenes.')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode, saving point clouds for visualization.')
    return parser



def main():
    args = get_argparse().parse_args()
    scene_root_path = args.scene_root_path
    scenes_list = os.listdir(scene_root_path)
    debug = args.debug
    # get all folder which include lang_feat_selected_imgs.json
    scenes_list = [scene for scene in scenes_list if os.path.isdir(os.path.join(scene_root_path, scene))] 
    scenes_list.sort()
    print(f"Found {len(scenes_list)} scenes.")

    for scene in tqdm(scenes_list):
        print("scene", scene)
        assert os.path.exists(os.path.join(scene_root_path, scene, 'visiable_gaussian_masks_per_frame_filtered.npy')), f"visiable_gaussian_masks_per_frame_filtered.npy not found in {scene}"

        visiable_gaussian_masks_per_frame_filtered_box_mask = np.load(os.path.join(scene_root_path, scene, 'visiable_gaussian_masks_per_frame_filtered.npy')) # NxS
        batch_size = min(args.batch_size, visiable_gaussian_masks_per_frame_filtered_box_mask.shape[0]-1)
        visiable_gaussian_masks_per_frame_filtered_box_mask_pair = np.zeros((visiable_gaussian_masks_per_frame_filtered_box_mask.shape[0], batch_size)).astype(np.int32) - 1
        for i in range(visiable_gaussian_masks_per_frame_filtered_box_mask.shape[0]):
            # we find top batch_size overlapped frames by check boolean mask overlap
            mask_i = visiable_gaussian_masks_per_frame_filtered_box_mask[i]
            overlap_list = []
            for j in range(visiable_gaussian_masks_per_frame_filtered_box_mask.shape[0]):
                if i == j:
                    continue
                mask_j = visiable_gaussian_masks_per_frame_filtered_box_mask[j]
                overlap = np.sum(mask_i & mask_j)
                overlap_list.append((j, overlap))
            
            overlap_list.sort(key=lambda x: x[1], reverse=True)
            topk = overlap_list[:batch_size]
            for k in range(len(topk)):
                visiable_gaussian_masks_per_frame_filtered_box_mask_pair[i, k] = topk[k][0]
                print(f"Frame {i} paired with Frame {topk[k][0]} with overlap {topk[k][1]}")

        np.save(os.path.join(scene_root_path, scene, f'visiable_gaussian_masks_per_frame_filtered_pair_top{batch_size}.npy'), visiable_gaussian_masks_per_frame_filtered_box_mask_pair)
        print("visiable_gaussian_masks_per_frame_filtered_box_mask_pair", visiable_gaussian_masks_per_frame_filtered_box_mask_pair.shape)
        print("visiable_gaussian_masks_per_frame_filtered_box_mask_pair example", visiable_gaussian_masks_per_frame_filtered_box_mask_pair[:5])



if __name__ == "__main__":
    main()
