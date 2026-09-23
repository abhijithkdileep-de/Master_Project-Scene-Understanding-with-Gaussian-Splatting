import numpy as np
import os 


import numpy as np
from scipy.spatial import cKDTree
import json
from tqdm import tqdm
import time 
import torch
from tqdm import trange

import open3d as o3d
import math

from plyfile import PlyData, PlyElement
from pathlib import Path


def save_ply(data_dict, file_path, max_sh_degree=3):
    """
    Save a 3dgs ply file.
      - f_rest channels (extra SH coefficients) are set to zero.
      - Before saving, the opacity and scale values are converted back to their raw forms,
        i.e. the inverse of the sigmoid (logit) and the inverse of exp (log) respectively.
      - The attributes are ordered as:
          x, y, z, nx, ny, nz,
          f_dc_0, f_dc_1, ..., f_dc_{C-1},
          f_rest_0, ..., f_rest_{R-1},
          opacity,
          scale_0, scale_1, ..., scale_{S-1},
          rot_0, rot_1, ..., rot_{Q-1}
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    N = data_dict["coord"].shape[0]

    # Coordinates and normals
    xyz = data_dict["coord"]  # (N, 3)
    normals = data_dict.get("normal", np.zeros_like(xyz))  # (N, 3)

    # f_dc channels from "color" (assumed shape: (N, 3))
    f_dc = data_dict["color"]
    # change rgb to dc
    C0 = 0.28209479177387814
    f_dc = (f_dc / 255. - 0.5) / C0
    print("f_dc range", f_dc.min(), f_dc.max())
    num_f_dc = f_dc.shape[1]

    # f_rest channels: set all to zero
    num_f_rest = 3 * (((max_sh_degree + 1) ** 2) - 1)  # For max_sh_degree=3, equals 45
    f_rest = np.zeros((N, num_f_rest), dtype=np.float32)

    # Inverse transform opacity:
    # data_dict["opacity"] was obtained via sigmoid: opacity = 1 / (1 + exp(-raw_opacity))
    # Inverse (logit): raw_opacity = ln(opacity/(1-opacity))
    opacity = data_dict["opacity"]
    if opacity.ndim == 1:
        opacity = opacity.reshape(-1, 1)
    eps = 1e-7  # to prevent division by zero
    opacity = np.clip(opacity, eps, 1 - eps)
    raw_opacity = np.log(opacity / (1 - opacity))

    # Inverse transform scales:
    # data_dict["scale"] was obtained via: scale = exp(raw_scale)
    # So, raw_scale = log(scale)
    scales = data_dict["scale"]
    raw_scales = np.log(scales)
    num_scale = scales.shape[1]

    # Rotation channels (quaternions)
    quat = data_dict["quat"]
    num_quat = quat.shape[1]

    # Build dtype list following the attribute order
    dtype_list = []
    # Coordinates and normals
    for attr in ["x", "y", "z", "nx", "ny", "nz"]:
        dtype_list.append((attr, "f4"))
    # f_dc channels
    for i in range(num_f_dc):
        dtype_list.append((f"f_dc_{i}", "f4"))
    # f_rest channels
    for i in range(num_f_rest):
        dtype_list.append((f"f_rest_{i}", "f4"))
    # Opacity (raw value)
    dtype_list.append(("opacity", "f4"))
    # Scale channels (raw values)
    for i in range(num_scale):
        dtype_list.append((f"scale_{i}", "f4"))
    # Rotation channels (quaternions)
    for i in range(num_quat):
        dtype_list.append((f"rot_{i}", "f4"))

    # Create and fill the structured array
    vertex_all = np.empty(N, dtype=dtype_list)
    vertex_all["x"] = xyz[:, 0]
    vertex_all["y"] = xyz[:, 1]
    vertex_all["z"] = xyz[:, 2]
    vertex_all["nx"] = normals[:, 0]
    vertex_all["ny"] = normals[:, 1]
    vertex_all["nz"] = normals[:, 2]
    for i in range(num_f_dc):
        vertex_all[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(num_f_rest):
        vertex_all[f"f_rest_{i}"] = f_rest[:, i]
    vertex_all["opacity"] = raw_opacity[:, 0]
    for i in range(num_scale):
        vertex_all[f"scale_{i}"] = raw_scales[:, i]
    for i in range(num_quat):
        vertex_all[f"rot_{i}"] = quat[:, i]

    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(file_path)
    print(f"Saved {file_path}")

# save gs ply instead of pc 

def save_views_to_ply( coords, color, scale, quat, opacity, filename):
    data_dict = {}
    # data_dict["coord"] = coords.cpu().numpy()
    # data_dict["color"] = ((color.cpu().numpy())+1)*127.5
    # data_dict["scale"] = scale.cpu().numpy()
    # data_dict["quat"] = quat.cpu().numpy()
    # data_dict["opacity"] = opacity.cpu().numpy()
    data_dict["coord"] = coords
    data_dict["color"] = color 
    data_dict["scale"] = scale
    data_dict["quat"] = quat
    data_dict["opacity"] = opacity

    save_ply(data_dict, filename)






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
    parser.add_argument('--augment_distance', type=float, default=0.3, help='Distance threshold for augmenting visibility.')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode, saving point clouds for visualization.')
    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--end_idx', default=-1, type=int)
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
    if args.end_idx == -1:
        scenes_list = scenes_list[args.start_idx :]
    else:
        scenes_list = scenes_list[args.start_idx : args.end_idx]
        

    for scene in tqdm(scenes_list):
        print("scene", scene)
        if os.path.exists(os.path.join(scene_root_path, scene, 'visiable_gaussian_masks_per_frame_filtered_box_mask.npy')):
            print(f"Scene {scene}: visiable_gaussian_masks_per_frame_filtered_box_mask.npy already exists, skip.")
            continue

        t0 = time.time()
        transformsfile = os.path.join(scene_root_path, scene, 'transforms_camera_positions_filtered.json')
        with open(transformsfile) as json_file:
            contents = json.load(json_file)
        chosen_contents = contents.copy() 

        visiable_gaussians_path = os.path.join(scene_root_path, scene, 'visiable_gaussian_masks_per_frame_filtered.npy')
        visiable_gaussians = np.load(visiable_gaussians_path) # MxN M is frame number, N is gaussian number
        frames = contents["frames"]

  
        # first we filter out the frames if it see to many gaussians:
        # we want < 204800 * 4


        # now we augment the visiable_gaussians, if one points (False) is cloes to visable in 1 meter, we set it to True
        coord_npy_path = os.path.join(scene_root_path, scene, 'coord.npy')
        coords = np.load(coord_npy_path)
        visiable_gaussians_box = np.zeros((visiable_gaussians.shape[0], 5)) # 2D center, width, height and rotation angle
        visiable_gaussians_box_mask = np.zeros_like(visiable_gaussians, dtype=bool)

        # we first debug and save the raw scene to ply

        for i in range(visiable_gaussians.shape[0]):
            visiable = np.where(visiable_gaussians[i])[0]
            visable_coords = coords[visiable]
            # denoise
            visable_coords_o3d = o3d.geometry.PointCloud()
            visable_coords_o3d.points = o3d.utility.Vector3dVector(visable_coords)
            cl, ind = visable_coords_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            visable_coords = np.asarray(cl.points)

            visable_coords_xy = visable_coords[:,:2]
            if len(visable_coords) < 10:
                print(f"Scene {scene}, frame {i}: only {len(visable_coords)} visiable points, skip.")
                continue
            # now we want to get a xy circule which have rotation and cover all points
            t1 = time.time()


            min_coords = np.min(visable_coords_xy, axis=0) # [min_x, min_y]
            max_coords = np.max(visable_coords_xy, axis=0) # [max_x, max_y]

            # 2. Calculate the Center (Midpoint of the bounding box diagonal)
            center = (min_coords + max_coords) / 2.0

            center_array = np.array(center).reshape(-1,2)
 
            center_z = np.mean(visable_coords[:,2])
            center_array_3d = np.array([center[0], center[1], center_z]).reshape(-1,3)

            # center_x, center_y, raidus = find_coarse_min_enclosing_circle(torch.tensor(visable_coords_xy).cuda())
            minimum_box_dict = exhaustive_rotated_bbox_search(
                visable_coords_xy, center_point=center, total_points=coords[:,:2]
            )
            

            visiable_gaussians_box_i = np.array([center_array[0,0] , center_array[0,1] , minimum_box_dict['width'] + args.augment_distance , minimum_box_dict['height'] + args.augment_distance , minimum_box_dict['best_angle_deg']])
            visiable_gaussians_box[i] = visiable_gaussians_box_i


            visiable_box = check_points_in_rotated_bbox(coords[:,:2], {'center': visiable_gaussians_box_i[:2], 'width': visiable_gaussians_box_i[2]+ args.augment_distance, 'height': visiable_gaussians_box_i[3]+ args.augment_distance, 'best_angle_deg': visiable_gaussians_box_i[4]})
            visiable_gaussians_box_mask[i]= visiable_box.astype(bool)
            # get new indices by calculating all points
            if args.debug:
                # save to ply for debug
                
                color_path = os.path.join(scene_root_path, scene, 'color.npy')
                colors = np.load(color_path)  
                """
                save to pc for fast debug
                """
                visiable_orig = np.where(visiable_gaussians[i])[0]
                if len(visiable_orig) == 0:
                    continue

                """
                save vis to pc for fast debug
                """
                pcd = o3d.geometry.PointCloud()
                # add center points to red
                # pcd.points = o3d.utility.Vector3dVector(coords[visiable_orig])
                # pcd.colors = o3d.utility.Vector3dVector(colors[visiable_orig])
                pcd.points = o3d.utility.Vector3dVector(np.vstack([coords[visiable_orig], center_array_3d]))
                colors_vis = colors[visiable_orig] / 255.
                colors_center = np.array([[1.0, 0.0, 0.0]])
                pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors_vis, colors_center]))
                o3d.io.write_point_cloud(os.path.join(scene_root_path, scene, f'selected_{i:03d}_orig_pc.ply'), pcd)
                

                quat = np.load(os.path.join(scene_root_path, scene, 'quat.npy'))
                scale = np.load(os.path.join(scene_root_path, scene, 'scale.npy'))
                opacity = np.load(os.path.join(scene_root_path, scene, 'opacity.npy'))
                save_views_to_ply(coords[visiable_orig], ((colors[visiable_orig])), scale[visiable_orig], quat[visiable_orig], opacity[visiable_orig], os.path.join(scene_root_path, scene, f'selected_{i:03d}_orig_gs.ply'))


                visiable_box = check_points_in_rotated_bbox(coords[:,:2], {'center': visiable_gaussians_box_i[:2], 'width': visiable_gaussians_box_i[2] + args.augment_distance, 'height': visiable_gaussians_box_i[3] + args.augment_distance, 'best_angle_deg': visiable_gaussians_box_i[4]})

                """
                save box visi to pc for fast debug
                """
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coords[visiable_box])
                pcd.colors = o3d.utility.Vector3dVector(colors[visiable_box] / 255.)
                o3d.io.write_point_cloud(os.path.join(scene_root_path, scene, f'selected_{i:03d}_box_pc.ply'), pcd)
                print("Save debug ply for frame", i, 'to', os.path.join(scene_root_path, scene, f'selected_{i:03d}_box_pc.ply'))

                save_views_to_ply(coords[visiable_box], ((colors[visiable_box])), scale[visiable_box], quat[visiable_box], opacity[visiable_box], os.path.join(scene_root_path, scene, f'selected_{i:03d}_box_gs.ply'))


        np.save(os.path.join(scene_root_path, scene, 'visiable_gaussian_masks_per_frame_filtered_box.npy'), visiable_gaussians_box_i)
        print("visiable_gaussians_box_i",  visiable_gaussians_box_i.shape)
        t2 = time.time()
        np.save(os.path.join(scene_root_path, scene, 'visiable_gaussian_masks_per_frame_filtered_box_mask.npy'), visiable_gaussians_box_mask)
        print("visiable_gaussians_box_mask", visiable_gaussians_box_mask.shape)
        print(f"Scene {scene}: augmenting visiable gaussians took {t2 - t1:.2f} seconds.")





if __name__ == "__main__":
    main()
