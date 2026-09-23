import os
import numpy as np

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class KITTI360GSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "segment",
        "instance",
        "quat",
        "scale",
        "opacity",
        "lang_feat",
        "lang_feat_index",
        "valid_feat_mask",
        "normal",
        "dino_feat",
    ]
    TARGET_FEATS = ["lang_feat"] # "dino_feat"
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment"]

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
                if asset[:-4] in self.TARGET_FEATS:
                    continue # skip loading target features during eval / test
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = (
                data_dict["coord"].astype(np.float16) / 10.0
            )  # apply scale

        if "pc_coord" in data_dict.keys():
            data_dict["pc_coord"] = data_dict["pc_coord"].astype(np.float16)

        if "pc_segment" in data_dict.keys():
            data_dict["pc_segment"] = data_dict["pc_segment"].astype(np.int32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float16)
            # print("color", data_dict["color"].shape)

        if "opacity" in data_dict.keys():
            data_dict["opacity"] = data_dict["opacity"].astype(np.float16).clip(0.001)
            data_dict["opacity"] = data_dict["opacity"].reshape(-1, 1)

        if "quat" in data_dict.keys():
            data_dict["quat"] = data_dict["quat"].astype(np.float16)

        if "sh" in data_dict.keys():
            data_dict["sh"] = data_dict["sh"].astype(np.float16)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float16)

        if "scale" in data_dict.keys():
            data_dict["scale"] = (data_dict["scale"].astype(np.float16) / 10.0).clip(
                0.01, 10.0
            )  # clip scale

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

        if "segment" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment").reshape([-1]).astype(np.int32)
            )

        return data_dict
