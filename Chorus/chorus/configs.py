from __future__ import annotations

import copy

CHECKPOINT_HUB = dict(repo_id="SceneSplatPro/Chorus", revision="main")

DEFAULT_CHECKPOINTS = {
    "chorus_3dgs": "lang-dino-enc-pretrain-scan-ppv2-mp3d-mcmc",
    "chorus_pts": "lang-dino-enc-pretrain-ppv2-mcmc-from-pts-params",
}

SUPPORTED_OUTPUTS = {"lang", "dino", "backbone_upcast", "backbone_last"}
TEACHER_OUTPUTS = {"lang", "dino"}


def _backbone(in_channels: int) -> dict:
    return dict(
        type="PT-v3m2",
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
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
    )


def _teachers() -> list[dict]:
    return [
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
            teacher_norm=dict(enabled=False),
        ),
    ]


def _outlier_filter() -> dict:
    return dict(
        enabled=True,
        method="mad",
        scale_mad_k=5.0,
        scale_threshold_factor=0.5,
        scale_max_real=10.0,
        scale_max_log=None,
        floater_candidate_quantile=0.995,
        floater_min_neighbors=3,
        floater_radius_mode="sqrt_scale",
        floater_radius_alpha=0.25,
        floater_radius_max=2.0,
        workers=1,
        verbose=True,
    )


def _preset(mode: str, feat_keys: tuple[str, ...], in_channels: int, raw_ply: bool) -> dict:
    grid_sample_keys = tuple(dict.fromkeys(feat_keys + ("coord", "normal")))
    collect_keys_test = ("coord", "grid_coord", "index")
    return dict(
        mode=mode,
        raw_ply=raw_ply,
        checkpoint=DEFAULT_CHECKPOINTS[mode],
        checkpoint_hub=copy.deepcopy(CHECKPOINT_HUB),
        feat_keys=feat_keys,
        model=dict(
            type="LangPretrainerMultiTeacher",
            backbone=_backbone(in_channels),
            projector_in_channels=1232,
            training_mode="joint",
            teachers=_teachers(),
        ),
        input_reader=dict(outlier_filter=_outlier_filter()),
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(type="Copy", keys_dict=dict(coord="origin_coord")),
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
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=grid_sample_keys,
                apply_to_pc=False,
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=collect_keys_test, feat_keys=feat_keys),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
        chunk_size=600_000,
        default_scene_name="awesome_scene",
    )


PRESETS = {
    "chorus_3dgs": _preset(
        "chorus_3dgs",
        feat_keys=("color", "opacity", "quat", "scale"),
        in_channels=11,
        raw_ply=True,
    ),
    "chorus_pts": _preset(
        "chorus_pts",
        feat_keys=("coord", "color", "normal"),
        in_channels=9,
        raw_ply=False,
    ),
}


def get_preset(mode: str) -> dict:
    try:
        return copy.deepcopy(PRESETS[mode])
    except KeyError as exc:
        raise KeyError(
            f"Unknown Chorus mode '{mode}'. Supported modes: {', '.join(sorted(PRESETS))}"
        ) from exc
