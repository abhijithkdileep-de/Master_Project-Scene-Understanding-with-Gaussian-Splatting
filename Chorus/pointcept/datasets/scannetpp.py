"""
ScanNet++ dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ScanNetPPDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "superpoint",
        "segment",
        "instance",
    ]

    def __init__(
        self,
        multilabel=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel

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
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "superpoint" in data_dict.keys():
            data_dict["superpoint"] = data_dict["superpoint"].astype(np.int32)

        if not self.multilabel:
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][:, 0].astype(np.int32)
            else:
                data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )

            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][:, 0].astype(np.int32)
            else:
                data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )
        else:
            raise NotImplementedError

        # Check for NaN or inf values before returning
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                has_nan = np.any(np.isnan(value))
                has_inf = np.any(np.isinf(value))
                
                if has_nan or has_inf:
                    error_msg = f"Invalid values detected in '{key}' for sample '{name}' (idx={idx}):\n"
                    if has_nan:
                        nan_count = np.sum(np.isnan(value))
                        error_msg += f"  - Found {nan_count} NaN values\n"
                    if has_inf:
                        inf_count = np.sum(np.isinf(value))
                        pos_inf_count = np.sum(np.isposinf(value))
                        neg_inf_count = np.sum(np.isneginf(value))
                        error_msg += f"  - Found {inf_count} inf values ({pos_inf_count} positive, {neg_inf_count} negative)\n"
                    error_msg += f"  - Array shape: {value.shape}, dtype: {value.dtype}\n"
                    error_msg += f"  - Data path: {data_path}"
                    
                    print(f"\n{error_msg}\n")
                    raise ValueError(error_msg)
        return data_dict