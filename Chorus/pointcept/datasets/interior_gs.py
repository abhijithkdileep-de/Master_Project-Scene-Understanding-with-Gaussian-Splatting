import os
import numpy as np

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset
import json
import random


@DATASETS.register_module()
class InteriorGSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "quat",
        "scale",
        "opacity",
        "instance",
    ]
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment", "pc_instance"]

    def __init__(
        self,
        multilabel=False,
        is_train=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel
        self.is_train = is_train

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if self.is_train:
                if asset[:-4] not in self.VALID_ASSETS:
                    continue
            else:
                if (
                    asset[:-4] not in self.VALID_ASSETS
                    and asset[:-4] not in self.EVAL_PC_ASSETS
                ):
                    continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float16)

        if "pc_coord" in data_dict.keys():
            data_dict["pc_coord"] = data_dict["pc_coord"].astype(np.float16)

        if "pc_segment" in data_dict.keys():
            data_dict["pc_segment"] = data_dict["pc_segment"].astype(np.int32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float16)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float16)

        if "opacity" in data_dict.keys():
            data_dict["opacity"] = data_dict["opacity"].astype(np.float16).clip(0.001)
            data_dict["opacity"] = data_dict["opacity"].reshape(-1, 1)

        if "quat" in data_dict.keys():
            data_dict["quat"] = data_dict["quat"].astype(np.float16)

        if "scale" in data_dict.keys():
            data_dict["scale"] = (
                data_dict["scale"].astype(np.float16).clip(1e-4, 5)
            )  # clip scale

        if "segment" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment").reshape([-1]).astype(np.int32)
            )
        
        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )

        return data_dict



def _resolve_render_dir(scene_path, render_dir_name="render_filtered"):
    candidates = []
    if render_dir_name:
        candidates.append(render_dir_name)
    candidates.extend(["render_filtered", "render"])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        render_dir = os.path.join(scene_path, candidate)
        if os.path.exists(render_dir):
            return render_dir
    raise FileNotFoundError(
        f"No render directory found under {scene_path}; tried {candidates}"
    )


def sample_intrinsic_and_extrinsic(
    scene_path,
    json_data,
    resize_w,
    resize_h,
    frames_pair,
    sample_num=1,
    valid_frames_idx=None,
    render_dir_name="render_filtered",
):
    # interiorgs

    frames_list = json_data.get("frames", [])
    frames_length = len(frames_list)
    # we sample one frame, and then sample (sample_num - 1) from the top-k pairs
    # first_idx = np.random.choice(frames_length, 1, replace=False)[0]
    if len(valid_frames_idx) == 0:
        valid_frames_idx = np.arange(frames_length) # all valid
    
    first_idx = np.random.choice(valid_frames_idx, 1, replace=False)[0]
    pair_indices = frames_pair[first_idx]  # get the paired frame indices
    if sample_num > 1:
        if len(pair_indices) >= (sample_num - 1):
            sampled_pair_indices = np.random.choice(pair_indices, sample_num - 1, replace=False)
        else:
            sampled_pair_indices = np.random.choice(pair_indices, sample_num - 1, replace=True)
        sample_idx = np.concatenate(([first_idx], sampled_pair_indices))
    else:
        sample_idx = np.array([first_idx])

    extrinsics = [] 
    images_names = []
    scene_image_path = _resolve_render_dir(scene_path, render_dir_name)


    fx_org = json_data.get("fx", 500)
    fy_org = json_data.get("fy", 500)
    w_org , h_org = json_data.get("resize", [640, 480])
    cx_org = json_data.get("cx", w_org / 2)
    cy_org = json_data.get("cy", h_org / 2)

    fl_x = fx_org * (resize_w / w_org)
    fl_y = fy_org * (resize_h / h_org)
    cx = cx_org * (resize_w / w_org)
    cy = cy_org * (resize_h / h_org)


    intrinsc_matrix = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    K = intrinsc_matrix.reshape(3, 3)
    # repeat sample num to Nx3x3
    K = np.repeat(K[None, :, :], sample_num, axis=0)

    for i in sample_idx:
        frame = frames_list[i]
        img_name = os.path.basename(frame["file_path"])
        img_path = os.path.join(scene_image_path, img_name)
        # img_name = 
        # img_name = 
        c2w = np.array(frame["transform_matrix"])   
        w2c = np.linalg.inv(c2w)       
        # random_frames_poses[i] = w2c.astype(np.float32)
        extrinsics.append(w2c.astype(np.float32))
        images_names.append(img_name)

    extrinsics = np.stack(extrinsics, axis=0).astype(np.float32) # Nx4x4

    return K, extrinsics, images_names, sample_idx



@DATASETS.register_module()
class Interior2DGSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "quat",
        "scale",
        "opacity",
        "visiable_gaussian_masks_per_frame_filtered_box_mask",
    ]
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment", "pc_instance"]

    def __init__(
        self,
        multilabel=False,
        is_train=True,
        frames_batch_size=1,
        dataset_type='interiorgs', # interiorgs, scannet, scannet_200, scannetpp, matterport
        resize_w=640,
        resize_h=480,
        maximal_gaussian_in_view=-1,
        render_dir_name="render_filtered",
        pair_top_k=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel
        self.is_train = is_train
        self.frames_batch_size = frames_batch_size
        # self.online_mode = online_mode
        self.dataset_type = dataset_type
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.maximal_gaussian_in_view = maximal_gaussian_in_view
        self.render_dir_name = render_dir_name
        self.pair_top_k = pair_top_k


    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if self.is_train:
                if asset[:-4] not in self.VALID_ASSETS:
                    continue
            else:
                if (
                    asset[:-4] not in self.VALID_ASSETS
                    and asset[:-4] not in self.EVAL_PC_ASSETS
                ):
                    continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float16)

        if "pc_coord" in data_dict.keys():
            data_dict["pc_coord"] = data_dict["pc_coord"].astype(np.float16)

        if "pc_segment" in data_dict.keys():
            data_dict["pc_segment"] = data_dict["pc_segment"].astype(np.int32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float16)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float16)

        if "opacity" in data_dict.keys():
            data_dict["opacity"] = data_dict["opacity"].astype(np.float16).clip(0.001)
            data_dict["opacity"] = data_dict["opacity"].reshape(-1, 1)

        if "quat" in data_dict.keys():
            data_dict["quat"] = data_dict["quat"].astype(np.float16)

        if "scale" in data_dict.keys():
            data_dict["scale"] = (
                data_dict["scale"].astype(np.float16).clip(1e-4, 5)
            )  # clip scale

        if "segment" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment").reshape([-1]).astype(np.int32)
            )
        
        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )

        # we read the visibility
        visibility_key = "visiable_gaussian_masks_per_frame_filtered_box_mask"
        if visibility_key in data_dict.keys():
            data_dict[visibility_key] = data_dict[visibility_key].astype(bool)

        else:
            raise NotImplementedError("visiable_gaussian_masks_per_frame_filtered_box_mask not found")

        if self.maximal_gaussian_in_view > 0:
            visible_counts = data_dict[visibility_key].sum(axis=1)
            valid_frames_idx = np.argwhere(visible_counts <= self.maximal_gaussian_in_view)
            if len(valid_frames_idx) == 0:
                print(
                    "Warning: all frames have too many gaussians in view, "
                    f"min is {visible_counts.min()}, selecting the least-dense frame"
                )
                least_idx = np.argmin(visible_counts)
                valid_frames_idx = np.array([[least_idx]])
                print("valid_frames_idx:", valid_frames_idx)
                valid_frames_idx = valid_frames_idx[:, 0]
            else:
                valid_frames_idx = valid_frames_idx[:, 0]

        else:
            valid_frames_idx = np.arange(data_dict[visibility_key].shape[0])

        resize_image_path = _resolve_render_dir(data_path, self.render_dir_name)

        json_path = os.path.join(data_path, "transforms_camera_positions_filtered.json")       
        pair_path = os.path.join(
            data_path,
            f"visiable_gaussian_masks_per_frame_filtered_pair_top{self.pair_top_k}.npy",
        )
        if not os.path.exists(pair_path):
            raise FileNotFoundError(f"Missing paired-frame file: {pair_path}")
        frames_pair = np.load(pair_path)
        
        with open(json_path, "r") as f:
            json_data = json.load(f)

        K, poses, images_names, sample_idx = sample_intrinsic_and_extrinsic(
            data_path,
            json_data,
            self.resize_w,
            self.resize_h,
            sample_num=self.frames_batch_size,
            frames_pair=frames_pair,
            valid_frames_idx=valid_frames_idx,
            render_dir_name=self.render_dir_name,
        )
        data_dict["K"] = K.astype(np.float32)[None, :, :, :] # 1xSx3x3

        # update the visiable_gaussian_masks_per_frame_filtered_box_mask with sample idx
        if visibility_key in data_dict.keys():
            data_dict[visibility_key] = data_dict[visibility_key][sample_idx]

        # we read resized images
        image_paths = []
        # image_arrays = []
        for i, img_name in enumerate(images_names):
            image_paths.append(os.path.join(resize_image_path, img_name))

        # each batch to make it clear we concat string to one string,separated by ;
        image_paths_concated = ';'.join(image_paths)
        data_dict["image_paths"] = image_paths_concated

           
        # we add visibility
        data_dict["poses"] = poses.astype(np.float32)[None, :, :, :]  # 1xSx4x4


        return data_dict
