_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/dataset/scannetpp.py",
]

# misc custom setting
debug = 0
gpu_nums = 1 #if debug else 4
batch_size = 3 * gpu_nums
batch_size_val = 1 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 8 * gpu_nums if not debug else 1
mix_prob = 0.0
empty_cache = False
enable_amp = True
test_only = False
clip_grad = 1.0
evaluate=False

resize_w=640
resize_h=480

# model settings
model = dict(
    type="LangPretrainerMultiTeacher2D",
    resize_w = resize_w,
    resize_h = resize_h,
    image_upsample_factor = 2,
    use_lora=True,
    online_image=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,  
    # Set with --options model.backbone_path=<released_chorus_checkpoint>.
    # Do not pass the base Chorus checkpoint through weight= for fresh adaptation.
    backbone_path=None,
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
        freeze_encoder=True,
    ),
    projector_in_channels=1232,
    training_mode="joint", # {'joint', 'alternating'}
    teachers=[
        dict(
            name="dino",
            target_key="dino_feat",
            mask_key=None,
            segment_key=None,
            light_projector=False,
            teacher_2D_model = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            downsample_ratio_2D = 2,
            projector=dict(
                out_channels=768,
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
epoch = 100 # 800 for full training, 400 for prototyping
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002/4, weight_decay=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002/4, 0.0002/4],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002/4)]

# dataset root
repo_root = "."
interiorgs_2d_data_root = (
    "data/interior_gs_2d_adaptation"
)

# training settings
feat_keys = ("color", "opacity", "quat", "scale")




grid_sample_keys = ( # control input entries for train / val
    "coord",
    "color",
    "normal",
    "opacity",
    "quat",
    "scale",
    "segment",
    # "lang_feat",
    # "dino_feat",
    # "valid_feat_mask",
)
grid_sample_keys_test = ( # control input entries for test
    "coord",
    "color",
    "normal",
    "opacity",
    "quat",
    "scale",
    "segment",
    "K",
    'poses',
    "image_paths",
)
collect_keys = (    # note, collect keys are gating
    "coord",
    "grid_coord",
    "segment",
    # "lang_feat",
    "dino_feat",
    "K",
    'poses',
    "image_paths",
    "pc_coord",
    "pc_segment",
)
collect_keys_test = ( 
    "coord",
    "grid_coord",
    "index",
    "segment",
    # "dino_feat",
    "K",
    'poses',
    "image_paths",
    "pc_coord",
    "pc_segment",
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None, save_best_for=["dino"]),
]

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type="Interior2DGSDataset",
        split="train",
        data_root=interiorgs_2d_data_root,
        frames_batch_size=4,
        render_dir_name="render_filtered",
        pair_top_k=4,
        sample_tail_classes=False,
        maximal_gaussian_in_view=204800*6,
        # skip_lang=True,
        transform=[
            dict(type="VisibleCrop"),
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
            dict(type="SphereCrop", point_max=204800*3, mode="center"),
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
        loop=1,
            ),
    val=dict(
            type="Interior2DGSDataset",
            split="test",
            data_root=interiorgs_2d_data_root,
            resize_w = resize_w,
            resize_h = resize_h,
            frames_batch_size=1,
            render_dir_name="render_filtered",
            pair_top_k=4,
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
            max_scenes=100, # max scenes to evaluate in a single val loader
        ),
    test=[],
)

# Tester
# dino_test_cfg=dict(
#                 name="dino",
#                 type="feature_similarity",
#                 target_key="dino_feat",
#                 sample_stride=4,
#                 chunk_size=200000,
#             )
test = ()
