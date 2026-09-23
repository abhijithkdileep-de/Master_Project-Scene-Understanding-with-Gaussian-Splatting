_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
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

# model settings
model = dict(
    type="LangPretrainerMultiTeacher",
    backbone=dict(
        type="PT-v3m2",
        in_channels=11,  # gaussian: color 3, quaternion 4, scale 3, opacity 1, w/o normal 3
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
            name="lang",
            target_key="lang_feat",
            mask_key="valid_feat_mask",
            segment_key="segment",
            projector=dict(
                out_channels=1152,
                depth=1,
                num_heads=32,
                patch_size=1024,
                drop_path_rate=0.1,
                block_type="mlp",
                input_norm=True,
                output_norm=False,
            ),
            criteria=[
                dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
                dict(type="SmoothL1Loss", reduction="mean", loss_weight=1.0),
                dict(
                    type="AggregatedContrastiveLoss",
                    temperature=0.2,
                    reduction="mean",
                    loss_weight=0.02,
                    schedule="all", # note we add pe teacher after training on lang+dino
                ),
            ],
            teacher_norm=dict(enabled=False),
        ),
        dict(
            name="dino",
            target_key="dino_feat",
            mask_key=None,
            segment_key=None,
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
        dict(
            name="pe_spatial",
            target_key="pe_feat",
            mask_key=None,
            segment_key="instance",
            l2_norm_pred=False,
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
                dict(
                    type="AggregatedContrastiveLoss",
                    temperature=0.2,
                    reduction="mean",
                    loss_weight=0.02,
                    schedule="last_80", # 'all', 'last_75', 'last_50', 'last_25'
                ),
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
interior_gs_root = (
    "/gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/interior_gs"
)

# training settings
feat_keys = ("color", "opacity", "quat", "scale")
grid_sample_keys = ( # control input entries for train / val
    "coord",
    "color",
    "opacity",
    "quat",
    "scale",
    "segment",
    "instance",
    "lang_feat",
    "dino_feat",
    "pe_feat",
    "valid_feat_mask",
)
grid_sample_keys_test = ( # control input entries for test
    "coord",
    "color",
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
    "instance",
    "lang_feat",
    "dino_feat",
    "pe_feat",
    "valid_feat_mask",
    "pc_coord",
    "pc_segment",
)
collect_keys_test = ( 
    "coord",
    "grid_coord",
    "index",
    "segment",
    # "dino_feat",
    "valid_feat_mask",
    "pc_coord",
    "pc_segment",
)
weight_pdnorm = {
    "ScanNetPPGS": 1,
    "ScanNetGS": 1,
    "Matterport3DGS": 1,
}  # first as main to iterate, ratio is multiplied to get total samples

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="LangPretrainMultiTeacherEval",
        teachers=[
            dict(
                name="lang",
                type="zero_shot",
                select_metric="fg_mIoU",
                class_names=f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt",
                text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2_so400m.pt",
                excluded_classes=["wall", "floor", "ceiling"],
                pred_label_mapping=None,
                ignore_index=-1,
                vote_k=25,
                enable_voting=True,
                confidence_threshold=0.1,
            ),
            dict(
                name="dino",
                type="feature_similarity",
                target_key="dino_feat",
                mask_key=None,
                mask_min_norm=0.05,
                sample_stride=4,
                chunk_size=200000,
                select_metric="cosine_similarity",
            ),
        ],
        evaluate_teachers=["lang"], # select teachers to evaluate
    ),
    dict(type="CheckpointSaver", save_freq=None, save_best_for=["lang"]),
    dict(
        type="PreciseEvaluator", test_last=False if not test_only else True
    ),  # use test_last=True to use current / loaded weight for evaluation
]

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type="ScanNetPPGSDataset",
        split=(
            "train_grid1.0cm_chunk6x6_stride4x4",
            "test_grid1.0cm_chunk6x6_stride4x4",
        ),
        data_root=scannetpp_data_root,
        sample_tail_classes=False,
        filtered_scene=["281ba69af1", "47b37eb6f9", "7e7cd69a59", "88627b561e", "578511c8a9", "ac48a9b736", "e9ac2fc517", '151178afd7', 'aee88e3a93', 'fde8a3b4c0', '0f25f24a4f', 'b24697b3a1', '7f68c514bd', '269fa95eb3', 'a56eda3549', '4e0b8cbd33', '4d451d9c36', '0fe8539661', 'c29b5e479c', '816e996553', '270ada6f0d', '654a4f341b', '82ff39b7ef', 'c601466b77', '124a6e789b', 'e3e0617f98', 'a72f31d6a6', '717485935', 'ac78eb8124', '2c7c10379b', 'c00b855082', '6248c6742d', '2b5ef64cad'],
        skip_pe=False,
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
            dict(type="GaussianNoiseInjection", noise_lr=1e-4, p=0.5),
            dict(type="GaussianScaleBlur", blur_factor_range=[1.0, 3.0], p=0.3),
            # dict(type="RandomJitter", sigma=0.005, clip=0.01),
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
    val=dict(
            type="ScanNetPPGSDataset",
            split=("val"),
            data_root=scannetpp_data_root,
            is_train=False,  # if not in train, we will load pc_segment and pc_coord if exsit, and not downsample them
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    return_grid_coord=True,
                ),
                # dict(type="SphereCrop", point_max=600000, mode="random"), # spconv limitation: int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, i.e., max 698k points for inference
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
            # max_scenes=100, # max scenes to evaluate in a single val loader
        ),
    test=[
        # scannet++
        dict(
            type="ScanNetPPGSDataset",
            split="val",
            data_root=scannetpp_data_root,
            is_train=False,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="NormalizeColor"),
                dict(
                    type="Copy",
                    keys_dict=dict(
                        segment="origin_segment",
                        coord="origin_coord",
                        valid_feat_mask="origin_feat_mask",
                        dino_feat="origin_dino_feat",
                        # pc_instance="origin_instance",
                    ),
                ),
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    apply_to_pc=False,
                    return_inverse=True,
                ),
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="test",
                    keys=grid_sample_keys_test,  # keep keys for inference is enough here
                    apply_to_pc=False,
                    return_grid_coord=True,
                ),
                crop=None,
                post_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=collect_keys_test,
                        feat_keys=feat_keys,
                    ),  # only keys for inference
                ],
                aug_transform=[
                    [
                        {
                            "type": "RandomRotateTargetAngle",
                            "angle": [0],
                            "axis": "z",
                            "center": [0, 0, 0],
                            "p": 1,
                        }
                    ]
                ],
            ),
        ),
        # scannet20
        dict(
            type="ScanNetGSDataset",
            split="val",
            data_root=scannet_data_root,
            is_train=False,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="NormalizeColor"),
                dict(
                    type="Copy",
                    keys_dict=dict(
                        segment="origin_segment",
                        coord="origin_coord",
                        valid_feat_mask="origin_feat_mask",
                        dino_feat="origin_dino_feat",
                        pc_instance="origin_instance",
                    ),
                ),
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    apply_to_pc=False,
                    return_inverse=True,
                ),
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="test",
                    keys=grid_sample_keys_test,  # keep keys for inference is enough here
                    apply_to_pc=False,
                    return_grid_coord=True,
                ),
                crop=None,
                post_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=collect_keys_test,
                        feat_keys=feat_keys,
                    ),  # only keys for inference
                ],
                aug_transform=[
                    [
                        {
                            "type": "RandomRotateTargetAngle",
                            "angle": [0],
                            "axis": "z",
                            "center": [0, 0, 0],
                            "p": 1,
                        }
                    ]
                ],
            ),
        ),
        # scannet200
        dict(
            type="ScanNet200GSDataset",
            split="val",
            data_root=scannet_data_root,
            is_train=False,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="NormalizeColor"),
                dict(
                    type="Copy",
                    keys_dict=dict(
                        segment="origin_segment",
                        coord="origin_coord",
                        valid_feat_mask="origin_feat_mask",
                        dino_feat="origin_dino_feat",
                        pc_instance="origin_instance",
                    ),
                ),
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    apply_to_pc=False,
                    return_inverse=True,
                ),
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="test",
                    keys=grid_sample_keys_test,  # keep keys for inference is enough here
                    apply_to_pc=False,
                    return_grid_coord=True,
                ),
                crop=None,
                post_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=collect_keys_test,
                        feat_keys=feat_keys,
                    ),  # only keys for inference
                ],
                aug_transform=[
                    [
                        {
                            "type": "RandomRotateTargetAngle",
                            "angle": [0],
                            "axis": "z",
                            "center": [0, 0, 0],
                            "p": 1,
                        }
                    ]
                ],
            ),
        ),
        # interior_gs_72
        dict(
            type="InteriorGSDataset",
            split="test",
            data_root=interior_gs_root,
            is_train=False,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="NormalizeColor"),
                dict(
                    type="Copy",
                    keys_dict=dict(
                        segment="origin_segment",
                        coord="origin_coord",
                        instance="origin_instance",
                    ),
                ),
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    apply_to_pc=False,
                    return_inverse=True,
                ),
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="test",
                    keys=grid_sample_keys_test,  # keep keys for inference is enough here
                    apply_to_pc=False,
                    return_grid_coord=True,
                ),
                crop=None,
                post_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=collect_keys_test,
                        feat_keys=feat_keys,
                    ),  # only keys for inference
                ],
                aug_transform=[
                    [
                        {
                            "type": "RandomRotateTargetAngle",
                            "angle": [0],
                            "axis": "z",
                            "center": [0, 0, 0],
                            "p": 1,
                        }
                    ]
                ],
            ),
        ),
        # matterport3d_160
        dict(
            type="Matterport3D_160_GSDataset",
            split="test_eval",
            data_root=matterport3d_data_root,
            is_train=False,
            skip_lang=True,
            skip_dino=True,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="NormalizeColor"),
                dict(
                    type="Copy",
                    keys_dict=dict(
                        segment="origin_segment",
                        coord="origin_coord",
                        # dino_feat="origin_dino_feat",
                        valid_feat_mask="origin_feat_mask",
                    ),
                ),
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    apply_to_pc=False,
                    return_inverse=True,
                ),
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="test",
                    keys=grid_sample_keys_test,  # keep keys for inference is enough here
                    apply_to_pc=False,
                    return_grid_coord=True,
                ),
                crop=None,
                limit_num=2000_000,
                post_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=collect_keys_test,
                        feat_keys=feat_keys,
                    ),  # only keys for inference
                ],
                aug_transform=[
                    [
                        {
                            "type": "RandomRotateTargetAngle",
                            "angle": [0],
                            "axis": "z",
                            "center": [0, 0, 0],
                            "p": 1,
                        }
                    ]
                ],
            ),
        ),
    ],
)

# Tester
dino_test_cfg=dict(
                name="dino",
                type="feature_similarity",
                target_key="dino_feat",
                sample_stride=4,
                chunk_size=200000,
            )
test = [
    # scannet++
    dict(
        type="LangPretrainMultiTeacherTester",
        verbose=True,
        teachers=[
            dict(
                name="lang",
                type="zero_shot",
                class_names=f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt",
                text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2_so400m.pt",
                excluded_classes=["wall", "floor", "ceiling"],
                enable_voting=True,
                vote_k=25,
                confidence_threshold=0.1,
                save_feat=False,
                skip_eval=False,
            ),
            dino_test_cfg,
        ],
        evaluate_teachers=["lang"],
    ),
    # scannet20
    dict(
        type="LangPretrainMultiTeacherTester",
        verbose=True,
        teachers=[
            dict(
                name="lang",
                type="zero_shot",
                select_metric="fg_mIoU",
                class_names=f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet20_labels.txt",
                text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet20_text_embeddings_siglip2_so400m.pt",
                excluded_classes=["wall", "floor", "ceiling"],
                enable_voting=True,
                vote_k=25,
                confidence_threshold=0.1,
                save_feat=False,
                skip_eval=False,
            ),
            dino_test_cfg,
        ],
        evaluate_teachers=["lang"],
    ),
    # scannet200
    dict(
        type="LangPretrainMultiTeacherTester",
        verbose=True,
        teachers=[
            dict(
                name="lang",
                type="zero_shot",
                select_metric="fg_mIoU",
                class_names=f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt",
                text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2_so400m.pt",
                excluded_classes=["wall", "floor", "ceiling"],
                enable_voting=True,
                vote_k=25,
                confidence_threshold=0.1,
                save_feat=False,
                skip_eval=False,
            ),
            dino_test_cfg,
        ],
        evaluate_teachers=["lang"],
    ),
    # interior_gs_72
    dict(
        type="LangPretrainMultiTeacherTester",
        verbose=True,
        teachers=[
            dict(
                name="lang",
                type="zero_shot",
                select_metric="fg_mIoU",
                class_names=f"{repo_root}/pointcept/datasets/preprocessing/interior_gs/metadata/semantic_labels.txt",
                text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/interior_gs/metadata/interior_gs_72_text_embeddings_siglip2-so400m.pt",
                excluded_classes=["wall", "floor", "ceiling"],
                enable_voting=True,
                vote_k=25,
                confidence_threshold=0.1,
                save_feat=False,
                skip_eval=False,
            ),
        ],
        evaluate_teachers=["lang"],
    ),
    # matterport3d_160
    dict(
        type="LangPretrainMultiTeacherTester",
        verbose=True,
        teachers=[
            dict(
                name="lang",
                type="zero_shot",
                select_metric="fg_mIoU",
                class_names=f"{repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/matterport_nyu160_labels.txt",
                text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/matterport-nyu160_text_embeddings_siglip2_so400m.pt",
                excluded_classes=["wall", "floor", "ceiling", "other furniture"],
                enable_voting=True,
                vote_k=25,
                confidence_threshold=0.1,
                save_feat=False,
                skip_eval=False,
            ),
            dino_test_cfg,
        ],
        evaluate_teachers=["lang"],
    )
]
