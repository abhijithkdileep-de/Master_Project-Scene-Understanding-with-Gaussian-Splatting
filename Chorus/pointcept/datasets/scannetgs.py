"""
ScanNet20 / ScanNet200 / GS Dataset
"""

import os
import numpy as np
import torch

from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class ScanNetGSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
        "quat",
        "scale",
        "opacity",
        "lang_feat",
        "lang_feat_index",
        "valid_feat_mask",
        "pc_instance",
        "dino_feat",
        "dino_feat_index",
        "pe_feat",
        "pe_feat_index",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)
    TARGET_FEATS = ["lang_feat"] # "dino_feat"
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment20"]

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        sample_tail=False,
        is_train=True,
        skip_lang=False,
        skip_dino=False,
        skip_pe=True,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        self.sample_tail = sample_tail
        self.is_train = is_train
        self.skip_lang = skip_lang
        self.skip_dino = skip_dino
        self.skip_pe = skip_pe
        super().__init__(**kwargs)

    def get_data_list(self, **kwargs):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

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
                # test / eval
                if (
                    asset[:-4] not in self.VALID_ASSETS
                    and asset[:-4] not in self.EVAL_PC_ASSETS
                ):
                    continue
                if asset[:-4] in self.TARGET_FEATS:
                    continue # skip loading target features during eval / test
            if self.skip_lang and asset[:-4] in [
                "lang_feat",
                "lang_feat_index",
                "valid_feat_mask",
            ]:
                continue
            if self.skip_dino and asset[:-4] in ["dino_feat", "dino_feat_index"]:
                continue
            if self.skip_pe and asset[:-4] in ["pe_feat", "pe_feat_index"]:
                continue
            try:
                data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
            except Exception as e:
                msg = (
                    f"\n🛑  Failed np.load() in ScanNetGSDataset\n"
                    f"    file   : {os.path.join(data_path, asset)}\n"
                    f"    scene  : {data_path}\n"
                    f"    reason : {e}\n"
                )
                print(msg, flush=True)
                raise RuntimeError(msg) from e
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float16)
        if "pc_coord" in data_dict.keys():
            data_dict["pc_coord"] = data_dict["pc_coord"].astype(np.float16)

        if "pc_segment200" in data_dict.keys():
            data_dict["pc_segment200"] = data_dict["pc_segment200"].astype(np.int32)
        if "pc_segment20" in data_dict.keys():
            data_dict["pc_segment20"] = data_dict["pc_segment20"].astype(np.int32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float16)
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float16)
        if "opacity" in data_dict.keys():
            data_dict["opacity"] = data_dict["opacity"].astype(np.float16)
            data_dict["opacity"] = data_dict["opacity"].reshape(-1, 1)
        if "quat" in data_dict.keys():
            data_dict["quat"] = data_dict["quat"].astype(np.float16)
        if "sh" in data_dict.keys():
            data_dict["sh"] = data_dict["sh"].astype(np.float16)
        if "scale" in data_dict.keys():
            data_dict["scale"] = (
                data_dict["scale"].astype(np.float16).clip(0, 1.5)
            )  # clip scale max to 1.5

        if "lang_feat" in data_dict.keys():
            data_dict["lang_feat"] = data_dict["lang_feat"].astype(np.float16)
            if (
                "lang_feat_index" in data_dict
                and "valid_feat_mask" in data_dict
                and "coord" in data_dict
            ):
                idx = data_dict["lang_feat_index"].astype(np.int64)
                feat = data_dict["lang_feat"]
                feat_dim = feat.shape[1]
                full = np.zeros((len(data_dict["coord"]), feat_dim), dtype=feat.dtype)
                full[idx] = feat
                data_dict["lang_feat"] = full
                data_dict.pop("lang_feat_index", None)
        if "valid_feat_mask" in data_dict.keys():
            data_dict["valid_feat_mask"] = data_dict["valid_feat_mask"].astype(bool)

        if "dino_feat" in data_dict.keys():
            data_dict["dino_feat"] = data_dict["dino_feat"].astype(np.float16)
            if ("dino_feat_index" in data_dict) and ("coord" in data_dict):
                idx = data_dict["dino_feat_index"].astype(np.int64)
                feat = data_dict["dino_feat"]
                feat_dim = feat.shape[1]
                full = np.zeros((len(data_dict["coord"]), feat_dim), dtype=feat.dtype)
                valid = (idx >= 0) & (idx < len(full))
                full[idx[valid]] = feat[valid]
                data_dict["dino_feat"] = full
                data_dict.pop("dino_feat_index", None)

        if "pe_feat" in data_dict.keys():
            data_dict["pe_feat"] = data_dict["pe_feat"].astype(np.float16)
            if ("pe_feat_index" in data_dict) and ("coord" in data_dict):
                idx = data_dict["pe_feat_index"].astype(np.int64)
                feat = data_dict["pe_feat"]
                feat_dim = feat.shape[1]
                full = np.zeros((len(data_dict["coord"]), feat_dim), dtype=feat.dtype)
                valid = (idx >= 0) & (idx < len(full))
                full[idx[valid]] = feat[valid]
                data_dict["pe_feat"] = full
                data_dict.pop("pe_feat_index", None)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "pc_segment20" in data_dict.keys():
            data_dict["pc_segment"] = (
                data_dict.pop("pc_segment20").reshape([-1]).astype(np.int32)
            )
        elif "pc_segment200" in data_dict.keys():
            data_dict["pc_segment"] = (
                data_dict.pop("pc_segment200").reshape([-1]).astype(np.int32)
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index

        # if self.sample_tail:
        #     # use data_dict["sampled_index"] to denote tail classes are sampled
        #     tail_mask = np.isin(data_dict["segment"], self.TAIL_CLASSES)
        #     data_dict["sampled_index"] = np.where(tail_mask)[0]

        return data_dict


@DATASETS.register_module()
class ScanNet200GSDataset(ScanNetGSDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
        "quat",
        "scale",
        "opacity",
        "lang_feat",
        "lang_feat_index",
        "valid_feat_mask",
        "pc_instance",
        "dino_feat",
        "dino_feat_index",
        "pe_feat",
        "pe_feat_index",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment200"]
