import os
import numpy as np

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class Matterport3DGSDataset(DefaultDataset):
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
        "dino_feat_index",
        "pe_feat",
        "pe_feat_index",
    ]
    TARGET_FEATS = ["lang_feat"] # "dino_feat"
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment"]

    def __init__(
        self,
        multilabel=False,
        is_train=True,
        skip_lang=False,
        skip_dino=False,
        skip_pe=True,
        filter_segment=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel
        self.is_train = is_train
        self.skip_lang = skip_lang
        self.skip_dino = skip_dino
        self.skip_pe = skip_pe
        self.filter_segment = filter_segment

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
                    f"\n🛑  Failed np.load() in Matterport3DGSDataset\n"
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

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float16)
        if "opacity" in data_dict.keys():
            data_dict["opacity"] = data_dict["opacity"].astype(np.float16)
            data_dict["opacity"] = data_dict["opacity"].reshape(-1, 1)
        if "quat" in data_dict.keys():
            data_dict["quat"] = data_dict["quat"].astype(np.float16)
        if "sh" in data_dict.keys():
            data_dict["sh"] = data_dict["sh"].astype(np.float16)
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float16)
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

            #  reconstruct to full length if compact
            if ("dino_feat_index" in data_dict) and ("coord" in data_dict):
                idx = data_dict["dino_feat_index"].astype(np.int64)
                feat = data_dict["dino_feat"]  # compact array [K, D]
                feat_dim = feat.shape[1] if feat.ndim == 2 else 1
                full = np.zeros((len(data_dict["coord"]), feat_dim), dtype=feat.dtype)
                valid = (idx >= 0) & (idx < len(full))
                if valid.any():
                    full[idx[valid]] = feat[valid]
                data_dict["dino_feat"] = full
                data_dict.pop("dino_feat_index", None)

        if "pe_feat" in data_dict.keys():
            data_dict["pe_feat"] = data_dict["pe_feat"].astype(np.float16)
            if ("pe_feat_index" in data_dict) and ("coord" in data_dict):
                idx = data_dict["pe_feat_index"].astype(np.int64)
                feat = data_dict["pe_feat"]
                feat_dim = feat.shape[1] if feat.ndim == 2 else 1
                full = np.zeros((len(data_dict["coord"]), feat_dim), dtype=feat.dtype)
                valid = (idx >= 0) & (idx < len(full))
                if valid.any():
                    full[idx[valid]] = feat[valid]
                data_dict["pe_feat"] = full
                data_dict.pop("pe_feat_index", None)

        if "segment" in data_dict.keys():
            # 21 classes processed by pointcept repo
            data_dict["segment"] = (
                data_dict.pop("segment").reshape([-1]).astype(np.int32)
            )
        elif "segment_nyu_160" in data_dict.keys():
            # top presented 160 nyu classes
            data_dict["segment"] = (
                data_dict.pop("segment_nyu_160").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "pc_segment" in data_dict.keys():
            data_dict["pc_segment"] = (
                data_dict.pop("pc_segment").reshape([-1]).astype(np.int32)
            )
        elif "pc_segment_nyu_160" in data_dict.keys():
            data_dict["pc_segment"] = (
                data_dict.pop("pc_segment_nyu_160").reshape([-1]).astype(np.int32)
            )

        if self.filter_segment:
            segment_mask = data_dict["segment"] >= 0
            for key in data_dict.keys():
                if len(data_dict[key]) == len(segment_mask) and "pc_" not in key:
                    data_dict[key] = data_dict[key][segment_mask]

        return data_dict


@DATASETS.register_module()
class Matterport3D_160_GSDataset(Matterport3DGSDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "segment_nyu_160",
        "instance",
        "quat",
        "scale",
        "opacity",
        "lang_feat",
        "lang_feat_index",
        "valid_feat_mask",
        "normal",
        "dino_feat",
        "dino_feat_index",
        "pe_feat",
        "pe_feat_index",
    ]
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment_nyu_160"]
