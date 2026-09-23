from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class Compose:
    def __init__(self, cfg=None):
        self.cfg = cfg or []
        self.transforms = [build_transform(item) for item in self.cfg]

    def __call__(self, data_dict):
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict


class Collect:
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.offset_keys = offset_keys_dict or dict(offset="coord")
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = {}
        for key in self.keys:
            if key in data_dict:
                data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


class Copy:
    def __init__(self, keys_dict=None):
        self.keys_dict = keys_dict or dict(coord="origin_coord", segment="origin_segment")

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if key not in data_dict:
                continue
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


class ToTensor:
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, str):
            return data
        if isinstance(data, int):
            return torch.LongTensor([data])
        if isinstance(data, float):
            return torch.FloatTensor([data])
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            if data.dtype == np.float64:
                return torch.from_numpy(data).float()
            return torch.from_numpy(data)
        if isinstance(data, Mapping):
            return {sub_key: self(item) for sub_key, item in data.items()}
        if isinstance(data, Sequence):
            return [self(item) for item in data]
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


class NormalizeColor:
    def __call__(self, data_dict):
        if "color" in data_dict:
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


class CenterShift:
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            return data_dict
        x_min, y_min, z_min = data_dict["coord"].min(axis=0)
        x_max, y_max, _ = data_dict["coord"].max(axis=0)
        if self.apply_z:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
        else:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
        data_dict["coord"] -= shift
        if "pc_coord" in data_dict:
            data_dict["pc_coord"] -= shift
        return data_dict


class RandomRotateTargetAngle:
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError

        dtype = data_dict["coord"].dtype
        center = self.center
        if center is None:
            coord_min = data_dict["coord"].min(axis=0)
            coord_max = data_dict["coord"].max(axis=0)
            center = (coord_min + coord_max) / 2
        data_dict["coord"] -= center
        data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t)).astype(dtype)
        data_dict["coord"] += center

        if "pc_coord" in data_dict:
            data_dict["pc_coord"] -= center
            data_dict["pc_coord"] = np.dot(data_dict["pc_coord"], np.transpose(rot_t))
            data_dict["pc_coord"] += center
        if "quat" in data_dict:
            quat_xyzw = np.roll(data_dict["quat"], shift=-1, axis=1)
            input_quat = R.from_quat(quat_xyzw)
            rot = R.from_matrix(rot_t)
            data_dict["quat"] = np.roll((rot * input_quat).as_quat(), shift=1, axis=1).astype(dtype)
        if "normal" in data_dict:
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


class GridSample:
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        importance_sample_key=None,
        apply_to_pc=True,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        if mode not in {"train", "test"}:
            raise ValueError("GridSample mode must be 'train' or 'test'")
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.importance_sample_key = importance_sample_key
        self.apply_to_pc = apply_to_pc

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            raise KeyError("GridSample requires 'coord'")
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        if "pc_coord" in data_dict and self.apply_to_pc:
            self._sample_pc_keys(data_dict)

        if self.mode == "train":
            if self.importance_sample_key is None:
                idx_select = (
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
                )
                idx_unique = idx_sort[idx_select]
            else:
                idx_unique = np.asarray(self.importance_sample(idx_sort, count, data_dict))
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                data_dict["displacement"] = self._displacement(
                    scaled_coord, grid_coord, data_dict
                )[idx_unique]
            for key in self.keys:
                if key in data_dict:
                    data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        data_part_list = []
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            data_part = dict(index=idx_part)
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_part["grid_coord"] = grid_coord[idx_part]
            if self.return_min_coord:
                data_part["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                data_dict["displacement"] = self._displacement(
                    scaled_coord, grid_coord, data_dict
                )[idx_part]
            for key in data_dict.keys():
                data_part[key] = data_dict[key][idx_part] if key in self.keys else data_dict[key]
            data_part_list.append(data_part)
        return data_part_list

    def _sample_pc_keys(self, data_dict):
        pc_coord = data_dict["pc_coord"]
        pc_grid_coord = np.floor(pc_coord / np.asarray(self.grid_size)).astype(int)
        pc_grid_coord -= pc_grid_coord.min(0)
        pc_key = self.hash(pc_grid_coord)
        pc_idx_sort = np.argsort(pc_key, kind="stable")
        pc_key_sorted = pc_key[pc_idx_sort]
        first_idx = np.nonzero(
            np.concatenate(([True], pc_key_sorted[1:] != pc_key_sorted[:-1]))
        )[0]
        pc_segment = data_dict.get("pc_segment")
        chosen_idx = []
        for i, start in enumerate(first_idx):
            end = first_idx[i + 1] if i + 1 < len(first_idx) else len(pc_idx_sort)
            cell_idx = pc_idx_sort[start:end]
            if pc_segment is not None:
                valid = cell_idx[pc_segment[cell_idx] != -1]
                chosen_idx.append(valid[0] if len(valid) else cell_idx[0])
            else:
                chosen_idx.append(cell_idx[0])
        chosen_idx = np.asarray(chosen_idx, dtype=np.int64)
        data_dict["pc_coord"] = data_dict["pc_coord"][chosen_idx]
        for key in ("pc_segment", "pc_instance"):
            if key in data_dict:
                data_dict[key] = data_dict[key][chosen_idx]

    def _displacement(self, scaled_coord, grid_coord, data_dict):
        displacement = scaled_coord - grid_coord - 0.5
        if self.project_displacement:
            displacement = np.sum(displacement * data_dict["normal"], axis=-1, keepdims=True)
        return displacement

    def importance_sample(self, idx_sort, count, data_dict):
        if isinstance(self.importance_sample_key, tuple):
            importance_sample = None
            for subkey in self.importance_sample_key:
                if "scale" in subkey and "scale" in data_dict:
                    mode = subkey.split("_")[1]
                    if mode == "max":
                        importance_attribute = np.max(data_dict["scale"], axis=-1)
                    elif mode == "prod":
                        importance_attribute = np.prod(data_dict["scale"], axis=-1)
                    elif mode == "min":
                        importance_attribute = np.min(data_dict["scale"], axis=-1)
                    else:
                        raise ValueError(f"Unsupported importance key {subkey}")
                else:
                    importance_attribute = data_dict[subkey]
                importance_sample = (
                    importance_attribute
                    if importance_sample is None
                    else importance_sample * importance_attribute
                )
        else:
            importance_sample = data_dict[self.importance_sample_key]
        grid_splits = np.cumsum(count[:-1])
        grid_indices_list = np.split(idx_sort, grid_splits)
        return [grid[importance_sample[grid].argmax()] for grid in grid_indices_list]

    @staticmethod
    def ravel_hash_vec(arr):
        if arr.ndim != 2:
            raise ValueError("ravel_hash_vec expects a 2D array")
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1
        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        if arr.ndim != 2:
            raise ValueError("fnv_hash_vec expects a 2D array")
        arr = arr.copy().astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


_TRANSFORMS = {
    "CenterShift": CenterShift,
    "Collect": Collect,
    "Copy": Copy,
    "GridSample": GridSample,
    "NormalizeColor": NormalizeColor,
    "RandomRotateTargetAngle": RandomRotateTargetAngle,
    "ToTensor": ToTensor,
}


def build_transform(cfg):
    cfg = dict(cfg)
    transform_type = cfg.pop("type")
    try:
        cls = _TRANSFORMS[transform_type]
    except KeyError as exc:
        raise KeyError(f"Unsupported package-mode transform: {transform_type}") from exc
    return cls(**cfg)
