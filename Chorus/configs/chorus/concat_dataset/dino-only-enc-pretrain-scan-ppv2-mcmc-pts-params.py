_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/dataset/scannetpp.py",
]

# misc custom setting
debug = 0
gpu_nums = 1 if debug else 4
batch_size = 2 * gpu_nums
batch_size_val = 1 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 8 * gpu_nums if not debug else 1
mix_prob = 0.0
empty_cache = False
enable_amp = True
test_only = False
clip_grad = 1.0
evaluate = False

# model settings
model = dict(
    type="LangPretrainerMultiTeacher",
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # gaussian: color 3, quaternion 4, scale 3, opacity 1, w/o normal 3
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),  # -> this direction
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
    ),
    projector_in_channels=1232,
    training_mode="joint", # {'joint', 'alternating'}
    teachers=[
        dict(
            name="dino",
            target_key="dino_feat",
            mask_key=None,
            segment_key="segment",
            projector=dict(
                out_channels=1024,
                depth=1,
                num_heads=16,
                patch_size=1024,
                drop_path_rate=0.1,
                block_type="mlp",
                input_norm=True,
                output_norm=False,
            ),
            criteria=[
                dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
                dict(type="SmoothL1Loss", reduction="mean", loss_weight=1.0),
            ],
            teacher_norm=dict(enabled=False),
        ),
    ],
)

# scheduler settings
epoch = 600
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset root
repo_root = "/home/yli7/projects/scene_3dgs_pro/scenesplat_pro"
scannet_data_root = "/gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs_lang_large"
scannetpp_data_root = (
    "/gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_3dgs_lang_large"
)
matterport3d_data_root = "/gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/matterport3d_scene_mcmc_3dgs_lang_large"

# training settings
feat_keys = ("coord", "color", "normal")
grid_sample_keys = ( # control input entries for train / val
    "coord",
    "color",
    "normal",
    "opacity",
    "quat",
    "scale",
    "segment",
    "lang_feat",
    "dino_feat",
    "valid_feat_mask",
)
grid_sample_keys_test = ( # control input entries for test
    "coord",
    "color",
    "normal",
    "opacity",
    "quat",
    "scale",
    "segment",
    "valid_feat_mask",
)
collect_keys = (    # note, collect keys are gating
    "coord",
    "grid_coord",
    "segment",
    # "lang_feat",
    "dino_feat",
    "valid_feat_mask",
    # "pc_coord",
    # "pc_segment",
)
collect_keys_test = ( 
    "coord",
    "grid_coord",
    "index",
    "segment",
    # "dino_feat",
    "valid_feat_mask",
    # "pc_coord",
    # "pc_segment",
)
weight_pdnorm = {
    "ScanNetPPGS": 2,
    "ScanNetGS": 1,
    "Matterport3DGS": 1,
}  # first as main to iterate, ratio is multiplied to get total samples

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=20, save_best_for=None),
]

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="ScanNetPPGSDataset",
                split=(
                    "train_grid1.0cm_chunk6x6_stride4x4",
                    "test_grid1.0cm_chunk6x6_stride4x4",
                ),
                data_root=scannetpp_data_root,
                sample_tail_classes=False,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.01),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=grid_sample_keys,
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=192000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=collect_keys,
                        feat_keys=feat_keys,
                    ),
                ],
                test_mode=False,
                loop=weight_pdnorm["ScanNetPPGS"],
                    ),
            dict(
                type="ScanNet200GSDataset",
                split=(
                    "train_grid1.0cm_chunk6x6_stride4x4",
                    "test_grid1.0cm_chunk6x6_stride4x4",
                ),
                data_root=scannet_data_root,
                sample_tail_classes=False,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.01),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=grid_sample_keys,
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=192000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=collect_keys,
                        feat_keys=feat_keys,
                    ),
                ],
                test_mode=False,
                loop=weight_pdnorm["ScanNetGS"],
                ),
        ]
    ),
)

# Tester
dino_test_cfg=dict(
                name="dino",
                type="feature_similarity",
                target_key="dino_feat",
                sample_stride=4,
                chunk_size=200000,
            )
test = ()
