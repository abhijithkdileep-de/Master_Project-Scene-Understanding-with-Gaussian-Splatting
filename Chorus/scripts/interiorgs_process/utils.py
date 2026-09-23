import torch
# import pycolmap_scene_manager as pycolmap
from typing import Literal
import numpy as np
from gsplat import rasterization

from plyfile import PlyData, PlyElement
import os 
import json
from pathlib import Path
from PIL import Image

def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1]*_features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(_features_rest.shape[1]*_features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    # l.append('language_feature')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l



def _detach_tensors_from_dict(d, inplace=True):
    if not inplace:
        d = d.copy()
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].detach()
    return d

def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1]*_features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(_features_rest.shape[1]*_features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    # l.append('language_feature')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_gsplat_dict_to_ply(
    ply_path: str,
    splats: dict,
):
    """
    Save the splats dictionary to a PLY file.
    Args:
        ply_path (str): Path to save the PLY file.
        splats (dict): Dictionary containing the splats data.
    """
    # Create a new PlyData object

    # mkdir_p(os.path.dirname(path))
    xyz = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = splats["features_dc"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = splats["features_rest"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = splats["opacity"].detach().cpu().numpy().reshape(-1, 1)
    scale = splats["scaling"].detach().cpu().numpy()
    rotation = splats["rotation"].detach().cpu().numpy()
    # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    # opacities = self._opacity.detach().cpu().numpy()
    # scale = self._scaling.detach().cpu().numpy()
    # rotation = self._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats["features_dc"], splats["features_rest"], splats["scaling"], splats["rotation"] )]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    # print("xyz shape", xyz.shape)
    # print("normals shape", normals.shape)
    # print("f_dc shape", f_dc.shape)
    # print("f_rest shape", f_rest.shape)
    # print("opacities shape", opacities.shape)
    # print("scale shape", scale.shape)
    # print("rotation shape", rotation.shape)

    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_path)


def load_ply(
    ply_path: str,
    data_dir: str,
    rasterizer: Literal["inria", "gsplat"] = "gsplat",
    max_sh_degree: int = 3,
    dataset: str = "scannetpp",
):
    plydata = PlyData.read(ply_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    splats = {
        "active_sh_degree": max_sh_degree,
        "means": torch.tensor(xyz).float().cuda(),
        "features_dc": torch.tensor(features_dc).float().cuda().transpose(1, 2).contiguous(),
        "features_rest": torch.tensor(features_extra).float().cuda().transpose(1, 2).contiguous(),
        "scaling": torch.tensor(scales).float().cuda(),
        "rotation": torch.tensor(rots).float().cuda(),
        "opacity": torch.tensor(opacities).float().cuda().squeeze(1),
    }

    # for key, value in splats.items():
    #     if key != "active_sh_degree":
    #         print("{}: {}".format(key, value.shape))
    #         print("min: {}, max: {}".format(value.min(), value.max()))
    _detach_tensors_from_dict(splats)
    if dataset == "scannetpp":
        # transform_json_file = os.path.join(data_dir, 'dslr', 'nerfstudio', 'lang_feat_selected_imgs.json')
        transform_json_file = os.path.join(data_dir, 'transforms_train_test.json')
    elif dataset == "scannet" or dataset == "holicity":
        # transform_json_file = os.path.join(data_dir, 'lang_feat_selected_imgs.json')
        transform_json_file = os.path.join(data_dir, 'transforms_train_test.json')
    elif dataset == "matterport":
        transform_json_file = os.path.join(data_dir, 'lang_feat_selected_imgs.json')
    elif dataset == "kitti360":
        transform_json_file = os.path.join(data_dir, 'meta.json')

    splats["transform_json_file"] = transform_json_file
    # splats["colmap_project"] = colmap_project
    # splats["colmap_dir"] = data_dir

    return splats

    # raise NotImplementedError("This is not used in the code, so not implemented")






def load_checkpoint(
    checkpoint: str,
    data_dir: str,
    rasterizer: Literal["inria", "gsplat"] = "gsplat",
    data_factor: int = 1,
):

    colmap_project = pycolmap.SceneManager(f"{data_dir}/sparse/0")
    colmap_project.load_cameras()
    colmap_project.load_images()
    colmap_project.load_points3D()
    model = torch.load(checkpoint)  # Make sure it is generated by 3DGS original repo
    if rasterizer == "inria":
        model_params, _ = model
        splats = {
            "active_sh_degree": model_params[0],
            "means": model_params[1],
            "features_dc": model_params[2],
            "features_rest": model_params[3],
            "scaling": model_params[4],
            "rotation": model_params[5],
            "opacity": model_params[6].squeeze(1),
        }
    elif rasterizer == "gsplat":

        model_params = model["splats"]
        splats = {
            "active_sh_degree": 3,
            "means": model_params["means"],
            "features_dc": model_params["sh0"],
            "features_rest": model_params["shN"],
            "scaling": model_params["scales"],
            "rotation": model_params["quats"],
            "opacity": model_params["opacities"],
        }
        # means: torch.Size([1000000, 3])
        # min: -12298.1533203125, max: 10229.1904296875
        # features_dc: torch.Size([1000000, 1, 3])
        # min: -2.661349058151245, max: 9.007434844970703
        # features_rest: torch.Size([1000000, 15, 3])
        # min: -1.0038543939590454, max: 1.0420862436294556
        # scaling: torch.Size([1000000, 3])
        # min: -20.09921646118164, max: 2.8412225246429443
        # rotation: torch.Size([1000000, 4])
        # min: -1.7583035230636597, max: 3.7463581562042236
        # opacity: torch.Size([1000000])
        # min: -12.972627639770508, max: 16.700193405151367
        # Total splats 1000000

        for key, value in splats.items():
            if key != "active_sh_degree":
                print("{}: {}".format(key, value.shape))
                print("min: {}, max: {}".format(value.min(), value.max()))
    else:
        raise ValueError("Invalid rasterizer")

    _detach_tensors_from_dict(splats)

    # Assuming only one camera
    for camera in colmap_project.cameras.values():
        camera_matrix = torch.tensor(
            [
                [camera.fx, 0, camera.cx],
                [0, camera.fy, camera.cy],
                [0, 0, 1],
            ]
        )
        break

    camera_matrix[:2, :3] /= data_factor

    splats["camera_matrix"] = camera_matrix
    splats["colmap_project"] = colmap_project
    splats["colmap_dir"] = data_dir

    return splats


def get_rpy_matrix(roll, pitch, yaw):
    roll_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll), np.cos(roll), 0],
            [0, 0, 0, 1.0],
        ]
    )

    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1.0],
        ]
    )
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1.0],
        ]
    )

    return yaw_matrix @ pitch_matrix @ roll_matrix


def get_viewmat_from_colmap_image(image):
    viewmat = torch.eye(4).float()  # .to(device)
    viewmat[:3, :3] = torch.tensor(image.R()).float()  # .to(device)
    viewmat[:3, 3] = torch.tensor(image.t).float()  # .to(device)
    return viewmat


def load_npy_to_gs(
    npy_path: str,
    rasterizer: Literal["inria", "gsplat"] = "gsplat",
    max_sh_degree: int = 0,
    dataset: str = "scannetpp",
):
    # plydata = PlyData.read(ply_path)
    coord = np.load(os.path.join(npy_path, "coord.npy"))  # (N, 3)
    color = np.load(os.path.join(npy_path, "color.npy"))  # (N, 3)
    quat = np.load(os.path.join(npy_path, "quat.npy"))  # (N, 4)
    scale = np.load(os.path.join(npy_path, "scale.npy"))  # (N, 3)
    opacities = np.load(os.path.join(npy_path, "opacity.npy"))  # (N, 1)
    # build gaussian
    xyz = coord.reshape(-1, 3)
    # opacities: already sigmoid, scale already go after exp
    # colors in 0 - 255, need to map to features_dc range
    # if "color" in attribute:
    #     C0 = 0.28209479177387814
    #     feature_pc = (feature_pc * C0).astype(np.float32) + 0.5
    #     feature_pc = np.clip(feature_pc, 0, 1)
    # # data = np.concatenate((data, feature_pc), axis=1)
    # data["color"] = feature_pc * 255
    features_dc = (color / 255.0 - 0.5) / 0.28209479177387814  # map to [-1.77, 1.77]
    features_dc = features_dc.reshape(-1, 3, 1)
    scales = scale.reshape(-1, 3)
    rots = quat.reshape(-1, 4)
    opacities = opacities.reshape(-1, 1)
    # features_extra = np.zeros((xyz.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))# sh degree 0, extra is 0 dimension

    splats = {
        "active_sh_degree": max_sh_degree,
        "means": torch.tensor(xyz).float().cuda(),
        "features_dc": torch.tensor(features_dc).float().cuda().transpose(1, 2).contiguous(),
        # "features_rest": torch.tensor(features_extra).float().cuda().transpose(1, 2).contiguous(),
        "scaling": torch.tensor(scales).float().cuda(),
        "rotation": torch.tensor(rots).float().cuda(),
        "opacity": torch.tensor(opacities).float().cuda().squeeze(1),
    }

    # for key, value in splats.items():
    #     if key != "active_sh_degree":
    #         print("{}: {}".format(key, value.shape))
    #         print("min: {}, max: {}".format(value.min(), value.max()))
    _detach_tensors_from_dict(splats)
    if dataset == "scannetpp":
        # transform_json_file = os.path.join(data_dir, 'dslr', 'nerfstudio', 'lang_feat_selected_imgs.json')
        transform_json_file = os.path.join(data_dir, 'transforms_train_test.json')
    elif dataset == "scannet" or dataset == "holicity":
        # transform_json_file = os.path.join(data_dir, 'lang_feat_selected_imgs.json')
        transform_json_file = os.path.join(data_dir, 'transforms_train_test.json')
    elif dataset == "matterport":
        transform_json_file = os.path.join(data_dir, 'lang_feat_selected_imgs.json')
    elif dataset == "kitti360":
        transform_json_file = os.path.join(data_dir, 'meta.json')
    elif dataset == "interiorgs":
        transform_json_file = os.path.join(npy_path, 'transforms_camera_positions.json')
    # transform_json_file = os.path.join(npy_path, 'lang_feat_selected_imgs.json')

    splats["transform_json_file"] = transform_json_file
    return splats



def camera_pose_to_visibility_mask_interiorgs(splats, gs_npy_path):
    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    # colors_rest = splats["features_rest"]
    colors = colors_dc #torch.cat([colors_dc, colors_rest], dim=1)

    # already in sigmoid
    opacities = (splats["opacity"])
    scales = (splats["scaling"])
    quats = splats["rotation"]
    # K = splats["camera_matrix"]
    colors.requires_grad = True
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device)
    transformsfile = splats["transform_json_file"]
    # first debug if color is correct or not
    colors.requires_grad = False
    # gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    # if debug:
    #     import trimesh             
    #     all_coord = means.detach().cpu().numpy()
    #     all_color = colors[:, 0, :].detach().cpu().numpy()
    #     all_color = (all_color * 0.28209479177387814 + 0.5).clip(0, 1)  # map back to [0, 1]
    #     all_color_np = all_color
    #     all_coord_np = all_coord
    #     point_cloud_all = trimesh.points.PointCloud(all_coord_np, all_color_np)
    #     point_cloud_all.export(f"./debug_scannet/debug_all_pointcloud.ply")


    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"] 
        cy = contents["cy"]

        # height = contents["h"] if 'h' in contents else contents["height"]
        # width = contents["w"] if 'w' in contents else contents["width"]
        width, height = contents['w'], contents['h']
        # resize = contents['resize']
        
        K = torch.tensor(
            [
                [focal_len_x, 0, cx],
                [0, focal_len_y, cy],
                [0, 0, 1],
            ]
        ).float()
        K = K.to('cuda')
        visiable_gaussian_masks = []

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            w2c = np.linalg.inv(c2w)
            R = w2c[:3,:3] #np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # image_path = os.path.join(image_path, f'{cam_name + extension}')
            # image_name = Path(cam_name).stem
            viewmat = torch.eye(4).float()  # .to(device)
            viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
            viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)
            viewmat = viewmat.to('cuda')
            ### color  debug
            # if debug:
                # print("viewmat[None]", viewmat[None])
            output, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors[:, 0, :],
                viewmats=viewmat[None],
                Ks=K[None],
                # sh_degree=0,
                width=width,
                height=height,
                render_mode='RGB+D'
            )

            output_debug = output.detach().cpu()[0].numpy() # H,W,3
            output_debug_depth = output_debug[..., 3]
            output_debug = output_debug[..., :3]
            C0 = 0.28209479177387814
            output_debug = (output_debug*C0).astype(np.float32) + 0.5
            output_debug = np.clip(output_debug, 0, 1)
            output_debug = (output_debug * 255).astype(np.uint8)
            output_debug_img = Image.fromarray(output_debug)
            output_debug_img.save(f"{gs_npy_path}/render/frame{frame_idx:06d}.png")
    
            output_debug_depth = output_debug_depth*10 # 
            output_debug_depth = output_debug_depth.clip(0, 255)
            output_debug_depth = output_debug_depth.astype(np.uint8)
            output_debug_depth_img = Image.fromarray(output_debug_depth)
            output_debug_depth_img.save(f"{gs_npy_path}/render/frame{frame_idx:06d}_depth.png")
            # get opacity by visbility
            colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device)
            colors_feats_0.requires_grad = True
            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats_0,
                viewmat[None],
                K[None],
                width=width,
                height=height,
            )  
            target_0 = (output_for_grad[0]).mean()

            target_0.backward()
            gaussian_denoms = colors_feats_0.grad[:, 0].detach().abs()  # (N,)
            print("gaussian_denoms shape", gaussian_denoms.shape, 'min', gaussian_denoms.min(), 'max', gaussian_denoms.max())

            gaussian_denoms_visable = gaussian_denoms[gaussian_denoms > 0]
            # print("gaussian_denoms_visable min", gaussian_denoms_visable.min())

            visiable_gaussian_mask = gaussian_denoms > 0 #1e-12 # ignore too small transimittence 
            # save visiable_gaussian_mask 
            visiable_gaussian_mask_np = visiable_gaussian_mask.detach().cpu().numpy()
            visiable_gaussian_masks.append(visiable_gaussian_mask_np)

            # gaussian_denoms > torch.quantile(gaussian_denoms_visable, 0.99) # we hope this will get ride of gaussian for occlusion
            # after > 0 (within the cone, we want to get percentile 95 value and filter again)
            # visiable_gaussian_mask = (gaussian_denoms > torch.quantile(gaussian_denoms, 0.95))
            # check if it is all False
            if torch.any(visiable_gaussian_mask):
                print("visiable_gaussian_mask sum", visiable_gaussian_mask.sum(), 'visable', gaussian_denoms_visable.shape[0], 'total', gaussian_denoms.shape[0])
    
            colors_feats_0.grad.zero_()
            # print(f"frame {frame_idx}, visiable_gaussian_mask num", visiable_gaussian_mask.sum(), "visable gaussian num", visiable_gaussian_mask.shape[0])

            # if debug:
            #     visable_coord = means[visiable_gaussian_mask]
            #     visable_color = colors[:, 0, :][visiable_gaussian_mask]
            #     visable_color = (visable_color * 0.28209479177387814 + 0.5).clamp(0, 1)  # map back to [0, 1]
            #     visable_color_np = visable_color.detach().cpu().numpy()
            #     visable_coord_np = visable_coord.detach().cpu().numpy()
            #     point_cloud = trimesh.points.PointCloud(visable_coord_np, visable_color_np)
            #     point_cloud.export(f"./debug_interiorgs/debug_visiable_pointcloud_{frame_idx}.ply")
            #     # save to ply and check it is align with 2D

            #     output, _, _ = rasterization(
            #         means[visiable_gaussian_mask],
            #         quats[visiable_gaussian_mask],
            #         scales[visiable_gaussian_mask],
            #         opacities[visiable_gaussian_mask],
            #         colors[:, 0, :][visiable_gaussian_mask],
            #         viewmats=viewmat[None],
            #         Ks=K[None],
            #         # sh_degree=0,
            #         width=width,
            #         height=height,
            #     )

            #     output_debug = output.detach().cpu()[0].numpy() # H,W,3
            #     C0 = 0.28209479177387814
            #     output_debug = (output_debug*C0).astype(np.float32) + 0.5
            #     output_debug = np.clip(output_debug, 0, 1)
            #     output_debug = (output_debug * 255).astype(np.uint8)
            #     output_debug_img = Image.fromarray(output_debug)
            #     output_debug_img.save(f"./debug_interiorgs/debug_output_{frame_idx}_visable.png")

            frame_idx += 1
        

        if len(visiable_gaussian_masks) == 0:
            print("No visiable gaussian masks found!")
            # save a empty array
            visiable_gaussian_masks = np.zeros((0, means.shape[0]), dtype=np.uint8)
            transformsfile_dir_path = Path(transformsfile).parent
            visiable_gaussian_mask_np_save_path = transformsfile_dir_path / "visiable_gaussian_masks_per_frame.npy"
            np.save(visiable_gaussian_mask_np_save_path, visiable_gaussian_masks)
            return
        visiable_gaussian_masks = np.stack(visiable_gaussian_masks, axis=0) # (F, N) F is number of frames, N is number of gaussian
        visiable_gaussian_masks = visiable_gaussian_masks.astype(np.uint8)
        transformsfile_dir_path = Path(transformsfile).parent
        visiable_gaussian_mask_np_save_path = transformsfile_dir_path / "visiable_gaussian_masks_per_frame.npy"

        np.save(visiable_gaussian_mask_np_save_path, visiable_gaussian_masks)



def prune_by_gradients_json(splats, inverse_extrinsics=True):

    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    # K = splats["camera_matrix"]
    colors.requires_grad = True
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device)

    transformsfile = splats["transform_json_file"]
    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"] 
        cy = contents["cy"]
        if "crop_edge" in contents:
            cx -= contents["crop_edge"]
            cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
            if "crop_edge" in contents:
                width -= 2*contents["crop_edge"]
                height -= 2*contents["crop_edge"]
        else:
            # if not specify, we assume the weight and height are twice the cx and cy
            width, height = cx * 2, cy * 2 
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            applied_transform = np.array([
                [0,  1,  0,  0],
                [1,  0,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1],
            ], dtype=float)
            c2w = np.dot(applied_transform, c2w)
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            if inverse_extrinsics: # some dataset save world-to-camera, some camera-to-world, careful!
                w2c = np.linalg.inv(c2w)
            else:
                w2c = c2w

            w2c[1:3] *= -1
            R = w2c[:3,:3] #np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # image_path = os.path.join(image_path, f'{cam_name + extension}')
            # image_name = Path(cam_name).stem
            viewmat = torch.eye(4).float()  # .to(device)
            viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
            viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)
            resize_ratio = 0.5
            fx_resize = focal_len_x * resize_ratio
            
            fy_resize = focal_len_y * resize_ratio
            cx_resize = cx * resize_ratio
            cy_resize = cy * resize_ratio
            
            K = torch.tensor(
                [
                    [fx_resize, 0, cx_resize],
                    [0, fy_resize, cy_resize],
                    [0, 0, 1],
                ]
            ).float()

            output, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors[:, 0, :],
                viewmats=viewmat[None],
                Ks=K[None],
                # sh_degree=3,
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
            )
            frame_idx += 1
            pseudo_loss = ((output.detach() + 1 - output) ** 2).mean()
            pseudo_loss.backward()
            # print(colors.grad.shape)
            gaussian_grads += (colors.grad[:, 0]).norm(dim=[1])
            colors.grad.zero_()

            # print("output shape", output.shape, "min", output.min(), "max", output.max())
        mask = gaussian_grads > 0
        print("Total splats", len(gaussian_grads))
        print("Pruned", (~mask).sum(), "splats")
        print("Remaining", mask.sum(), "splats")
        splats = splats.copy()
        splats["means"] = splats["means"][mask]
        splats["features_dc"] = splats["features_dc"][mask]
        splats["features_rest"] = splats["features_rest"][mask]
        splats["scaling"] = splats["scaling"][mask]
        splats["rotation"] = splats["rotation"][mask]
        splats["opacity"] = splats["opacity"][mask]
    
    return splats


def prune_by_gradients_opencv(splats, inverse_extrinsics=True):

    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    # K = splats["camera_matrix"]
    colors.requires_grad = True
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device)

    transformsfile = splats["transform_json_file"]
    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"] 
        cy = contents["cy"]
        if "crop_edge" in contents:
            cx -= contents["crop_edge"]
            cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
            if "crop_edge" in contents:
                width -= 2*contents["crop_edge"]
                height -= 2*contents["crop_edge"]
        else:
            # if not specify, we assume the weight and height are twice the cx and cy
            width, height = cx * 2, cy * 2 
        frames = contents["frames"]


        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            # some dataset save world-to-camera, some camera-to-world, careful!
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = w2c[:3,:3]  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            viewmat = torch.eye(4).float()  # .to(device)
            viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
            viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)

            # resize_ratio = 0.5
            # fx_resize = focal_len_x * resize_ratio
            
            # fy_resize = focal_len_y * resize_ratio
            # cx_resize = cx * resize_ratio
            # cy_resize = cy * resize_ratio
            
            K = torch.tensor(
                [
                    [focal_len_x, 0, cx],
                    [0, focal_len_y, cy],
                    [0, 0, 1],
                ]
            ).float()

            output, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors[:, 0, :],
                viewmats=viewmat[None],
                Ks=K[None],
                # sh_degree=3,
                width=width,
                height=height,
            )
            frame_idx += 1
            pseudo_loss = ((output.detach() + 1 - output) ** 2).mean()
            pseudo_loss.backward()
            # print(colors.grad.shape)
            gaussian_grads += (colors.grad[:, 0]).norm(dim=[1])
            colors.grad.zero_()

            # debug
            # output_debug = output.detach().cpu()[0].numpy() # H,W,3
            # C0 = 0.28209479177387814
            # output_debug = (output_debug*C0).astype(np.float32) + 0.5
            # output_debug = np.clip(output_debug, 0, 1)
            # output_debug = (output_debug * 255).astype(np.uint8)
            # output_debug_img = Image.fromarray(output_debug)
            # output_debug_img.save(f"./debug/scannet_debug_output_{frame_idx}.png")

            # raise NotImplementedError("Debugging, remove this line to continue")

            # print("output shape", output.shape, "min", output.min(), "max", output.max())
        mask = gaussian_grads > 0
        print("Total splats", len(gaussian_grads))
        print("Pruned", (~mask).sum(), "splats")
        print("Remaining", mask.sum(), "splats")
        splats = splats.copy()
        splats["means"] = splats["means"][mask]
        splats["features_dc"] = splats["features_dc"][mask]
        splats["features_rest"] = splats["features_rest"][mask]
        splats["scaling"] = splats["scaling"][mask]
        splats["rotation"] = splats["rotation"][mask]
        splats["opacity"] = splats["opacity"][mask]
    
    return splats




def prune_by_gradients(splats):
    colmap_project = splats["colmap_project"]
    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    colors.requires_grad = True
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device)
    for image in sorted(colmap_project.images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors[:, 0, :],
            viewmats=viewmat[None],
            Ks=K[None],
            sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )
        frame_idx += 1
        pseudo_loss = ((output.detach() + 1 - output) ** 2).mean()
        pseudo_loss.backward()
        # print(colors.grad.shape)
        gaussian_grads += (colors.grad[:, 0]).norm(dim=[1])
        colors.grad.zero_()

    mask = gaussian_grads > 0
    print("Total splats", len(gaussian_grads))
    print("Pruned", (~mask).sum(), "splats")
    print("Remaining", mask.sum(), "splats")
    splats = splats.copy()
    splats["means"] = splats["means"][mask]
    splats["features_dc"] = splats["features_dc"][mask]
    splats["features_rest"] = splats["features_rest"][mask]
    splats["scaling"] = splats["scaling"][mask]
    splats["rotation"] = splats["rotation"][mask]
    splats["opacity"] = splats["opacity"][mask]
    return splats


def create_checkerboard(width, height, size=64):
    checkerboard = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, size):
        for x in range(0, width, size):
            if (x // size + y // size) % 2 == 0:
                checkerboard[y : y + size, x : x + size] = 255
            else:
                checkerboard[y : y + size, x : x + size] = 128
    return checkerboard


def torch_to_cv(tensor, permute=False):
    if permute:
        tensor = torch.clamp(tensor.permute(1, 2, 0), 0, 1).cpu().numpy()
    else:
        tensor = torch.clamp(tensor, 0, 1).cpu().numpy()
    return (tensor * 255).astype(np.uint8)[..., ::-1]

def test_proper_pruning(splats, splats_after_pruning):
    colmap_project = splats["colmap_project"]
    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]

    means_pruned = splats_after_pruning["means"]
    colors_dc_pruned = splats_after_pruning["features_dc"]
    colors_rest_pruned = splats_after_pruning["features_rest"]
    colors_pruned = torch.cat([colors_dc_pruned, colors_rest_pruned], dim=1)
    opacities_pruned = torch.sigmoid(splats_after_pruning["opacity"])
    scales_pruned = torch.exp(splats_after_pruning["scaling"])
    quats_pruned = splats_after_pruning["rotation"]

    K = splats["camera_matrix"]
    total_error = 0
    max_pixel_error = 0
    for image in sorted(colmap_project.images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )

        output_pruned, _, _ = rasterization(
            means_pruned,
            quats_pruned,
            scales_pruned,
            opacities_pruned,
            colors_pruned,
            viewmats=viewmat[None],
            Ks=K[None],
            sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )

        total_error += torch.abs((output - output_pruned)).sum()
        max_pixel_error = max(
            max_pixel_error, torch.abs((output - output_pruned)).max()
        )

    percentage_pruned = (
        (len(splats["means"]) - len(splats_after_pruning["means"]))
        / len(splats["means"])
        * 100
    )

    assert max_pixel_error < 1 / (
        255 * 2
    ), "Max pixel error should be less than 1/(255*2), safety margin"
    print(
        "Report {}% pruned, max pixel error = {}, total pixel error = {}".format(
            percentage_pruned, max_pixel_error, total_error
        )
    )


def test_proper_pruning_json(splats, splats_after_pruning, inverse_extrinsics=True):
    # colmap_project = splats["colmap_project"]
    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]

    means_pruned = splats_after_pruning["means"]
    colors_dc_pruned = splats_after_pruning["features_dc"]
    colors_rest_pruned = splats_after_pruning["features_rest"]
    colors_pruned = torch.cat([colors_dc_pruned, colors_rest_pruned], dim=1)
    opacities_pruned = torch.sigmoid(splats_after_pruning["opacity"])
    scales_pruned = torch.exp(splats_after_pruning["scaling"])
    quats_pruned = splats_after_pruning["rotation"]

    # K = splats["camera_matrix"]
    total_error = 0
    max_pixel_error = 0
    transformsfile = splats["transform_json_file"]

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"] 
        cy = contents["cy"]
        if "crop_edge" in contents:
            cx -= contents["crop_edge"]
            cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
            if "crop_edge" in contents:
                width -= 2*contents["crop_edge"]
                height -= 2*contents["crop_edge"]
        else:
            # if not specify, we assume the weight and height are twice the cx and cy
            width, height = cx * 2, cy * 2 
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            applied_transform = np.array([
                [0,  1,  0,  0],
                [1,  0,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1],
            ], dtype=float)
            c2w = np.dot(applied_transform, c2w)
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            if inverse_extrinsics: # some dataset save world-to-camera, some camera-to-world, careful!
                w2c = np.linalg.inv(c2w)
            else:
                w2c = c2w

            w2c[1:3] *= -1
            R = w2c[:3,:3] #np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # image_path = os.path.join(image_path, f'{cam_name + extension}')
            # image_name = Path(cam_name).stem
            viewmat = torch.eye(4).float()  # .to(device)
            viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
            viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)
            resize_ratio = 0.5
            fx_resize = focal_len_x * resize_ratio
            
            fy_resize = focal_len_y * resize_ratio
            cx_resize = cx * resize_ratio
            cy_resize = cy * resize_ratio
            
            K = torch.tensor(
                [
                    [fx_resize, 0, cx_resize],
                    [0, fy_resize, cy_resize],
                    [0, 0, 1],
                ]
            ).float()

            output, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats=viewmat[None],
                Ks=K[None],
                sh_degree=3,
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
            )

            output_pruned, _, _ = rasterization(
                means_pruned,
                quats_pruned,
                scales_pruned,
                opacities_pruned,
                colors_pruned,
                viewmats=viewmat[None],
                Ks=K[None],
                sh_degree=3,
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
            )

            frame_idx += 1

            total_error += torch.abs((output - output_pruned)).sum()
            max_pixel_error = max(
                max_pixel_error, torch.abs((output - output_pruned)).max()
            )

        percentage_pruned = (
            (len(splats["means"]) - len(splats_after_pruning["means"]))
            / len(splats["means"])
            * 100
        )

        assert max_pixel_error < 1 / (
            255 * 2
        ), f"Max pixel error {max_pixel_error} should be less than 1/(255*2), safety margin"
        print(
            "Report {}% pruned, max pixel error = {}, total pixel error = {}".format(
                percentage_pruned, max_pixel_error, total_error
            )
        )




def test_proper_pruning_opencv(splats, splats_after_pruning, inverse_extrinsics=True):
    # colmap_project = splats["colmap_project"]
    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]

    means_pruned = splats_after_pruning["means"]
    colors_dc_pruned = splats_after_pruning["features_dc"]
    colors_rest_pruned = splats_after_pruning["features_rest"]
    colors_pruned = torch.cat([colors_dc_pruned, colors_rest_pruned], dim=1)
    opacities_pruned = torch.sigmoid(splats_after_pruning["opacity"])
    scales_pruned = torch.exp(splats_after_pruning["scaling"])
    quats_pruned = splats_after_pruning["rotation"]

    # K = splats["camera_matrix"]
    total_error = 0
    max_pixel_error = 0
    transformsfile = splats["transform_json_file"]

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"] 
        cy = contents["cy"]
        if "crop_edge" in contents:
            cx -= contents["crop_edge"]
            cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
            if "crop_edge" in contents:
                width -= 2*contents["crop_edge"]
                height -= 2*contents["crop_edge"]
        else:
            # if not specify, we assume the weight and height are twice the cx and cy
            width, height = cx * 2, cy * 2 
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            # some dataset save world-to-camera, some camera-to-world, careful!
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = w2c[:3,:3]  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            viewmat = torch.eye(4).float()  # .to(device)
            viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
            viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)
            # image_path = os.path.join(image_path, f'{cam_name + extension}')
            # image_name = Path(cam_name).stem
            # resize_ratio = resize[1] / (640 - 2 * contents["crop_edge"]) if "crop_edge" in contents else 0
            # fx_resize = focal_len_x * resize_ratio
            
            # fy_resize = focal_len_y * resize_ratio
            # cx_resize = cx * resize_ratio
            # cy_resize = cy * resize_ratio


            K = torch.tensor(
                [
                    [focal_len_x, 0, cx],
                    [0, focal_len_y, cy],
                    [0, 0, 1],
                ]
            ).float()

            output, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats=viewmat[None],
                Ks=K[None],
                sh_degree=3,
                width=width,
                height=height,
            )

            output_pruned, _, _ = rasterization(
                means_pruned,
                quats_pruned,
                scales_pruned,
                opacities_pruned,
                colors_pruned,
                viewmats=viewmat[None],
                Ks=K[None],
                sh_degree=3,
                width=width,
                height=height,
            )

            frame_idx += 1

            total_error += torch.abs((output - output_pruned)).sum()
            max_pixel_error = max(
                max_pixel_error, torch.abs((output - output_pruned)).max()
            )

        percentage_pruned = (
            (len(splats["means"]) - len(splats_after_pruning["means"]))
            / len(splats["means"])
            * 100
        )

        assert max_pixel_error < 1 / (
            255 * 2
        ), "Max pixel error should be less than 1/(255*2), safety margin"
        print(
            "Report {}% pruned, max pixel error = {}, total pixel error = {}".format(
                percentage_pruned, max_pixel_error, total_error
            )
        )

