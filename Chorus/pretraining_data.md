# Chorus Pretraining Data

Chorus pretraining uses preprocessed per-scene `*.npy` assets with compact teacher features. For each scene, the expected files are:

```bash
<data_root>/<split>/<scene_id>/
├── coord.npy                 # N x 3 float16 Gaussian centers
├── color.npy                 # N x 3 uint8 RGB, 0..255
├── opacity.npy               # N float16 sigmoid opacity, [0, 1]
├── scale.npy                 # N x 3 float16 positive Gaussian scales
├── quat.npy                  # N x 4 float16 unit quaternion in wxyz order
├── normal.npy                # N x 3 float16 normals; required by chorus point-cloud variant
├── segment.npy               # N-row semantic labels
├── lang_feat.npy             # K x 1152 float16 compact SigLIP2 teacher features
├── lang_feat_index.npy       # K int32 original Gaussian row indices for lang_feat
├── dino_feat.npy             # Kd x 1024 float16 compact DINOv3 teacher features
├── dino_feat_index.npy       # Kd int32 original Gaussian row indices for dino_feat
├── pe_feat.npy               # optional Kp x 1024 float16 compact PE-Spatial features
├── pe_feat_index.npy         # optional Kp int32 original Gaussian row indices for pe_feat
└── valid_feat_mask.npy       # bool mask for valid teacher-feature rows
```

The teacher features are compact. `*_feat.npy` stores only rows that received a valid teacher feature, and `*_feat_index.npy` maps those rows back to the original Gaussian row ids. The Chorus loaders expand compact features at load time, and the chunking script keeps the compact sidecars when it writes chunk folders.

The data preparation code has three separate repositories:

- [chorus_data_generator](https://github.com/unique1i/chorus_data_generator)
    - `preprocess_2d_language_feature/`
    - `occamlgs_dev/`
    - `ludvig_dev/`

Each data-preparation folder has its own README with environment setup and example commands. This page ties the pieces together into the Chorus pretraining data recipe.

## Source RGB/Cameras

The data-preparation repos do not download the original RGB datasets. Before running the feature-preparation steps, keep the source RGB/camera folders in the layout expected by the scripts, and place the released view-list JSONs in the scene folders.

The selected RGB view JSONs used for Chorus are in [3dgs_training_views](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k/tree/main/3dgs_training_views). If you are reproducing the pretraining data, use those `lang_feat_selected_imgs.json` files instead of reselecting frames.

For ScanNet, `process_dataset.py` looks under both `scans` and `scans_test`:

```text
<scannet_root>/
├── scans/<scene>/
│   ├── color_interval/
│   ├── pose/
│   ├── intrinsic/
│   └── lang_feat_selected_imgs.json
└── scans_test/<scene>/
    ├── color_interval/
    ├── pose/
    ├── intrinsic/
    └── lang_feat_selected_imgs.json
```

For ScanNet++, it looks under both `data` and `sem_test`:

```text
<scannetpp_root>/
├── data/<scene>/dslr/
│   ├── undistorted_images/
│   └── nerfstudio/
│       └── lang_feat_selected_imgs.json
└── sem_test/<scene>/dslr/
    ├── undistorted_images/
    └── nerfstudio/
        └── lang_feat_selected_imgs.json
```

For Matterport3D, you can download the scene-level images folder from the [OpenScene](https://github.com/pengsongyou/openscene) Matterport download, using the `matterport_2d` option. The expected layout is:

```text
<matterport_2d_root>/<scene>/
├── color/
├── depth/
├── pose/
├── intrinsic/
├── lang_feat_selected_imgs.json
└── transforms_train.json
```

For Matterport3D scene-level preparation, use `3dgs_training_views/matterport3d/scene/<scene>/lang_feat_selected_imgs.json`. The `matterport3d/region/` view JSONs are for region-level evaluation metadata, not this scene-level feature-preparation step.

## Starting Point

Download the released base 3DGS datasets:

- ScanNet: [scannet_mcmc_3dgs_lang_base](https://huggingface.co/datasets/clapfor/scannet_mcmc_3dgs_lang_base)
- ScanNet++ v2: [scannetpp_v2_mcmc_3dgs_lang_base](https://huggingface.co/datasets/clapfor/scannetpp_v2_mcmc_3dgs_lang_base)
- Matterport3D scene-level: [matterport3d_scene_mcmc_3dgs_lang_base](https://huggingface.co/datasets/clapfor/matterport3d_scene_mcmc_3dgs_lang_base)

These Hugging Face repos are gated, so request access and log in before downloading. For this recipe, you only need the scene-level base splats. The helper below skips the pre-chunked `*_grid...` folders and the SceneSplat language-feature sidecars (`valid_feat_mask.npy`, `lang_feat.npy`, and `lang_feat_index.npy`), because Chorus requires new files.

```bash
python scripts/data/download_base_3dgs.py --output-root /path/to/base_3dgs_releases
```

The helper writes folders named after the Hugging Face repos, for example:

```text
/path/to/base_3dgs_releases/scannet_mcmc_3dgs_lang_base
/path/to/base_3dgs_releases/scannetpp_v2_mcmc_3dgs_lang_base
/path/to/base_3dgs_releases/matterport3d_scene_mcmc_3dgs_lang_base
```

Please rename them to `_lang_large` before adding Chorus features. The examples below use `_lang_large` as the final root name.

For Matterport3D, the same download keeps the released `test_eval` folder. That folder is the region-level evaluation set used with `regions_test.txt`, it is not part of the scene-level teacher-feature preparation.

Those datasets already provide the processed per-scene Gaussian arrays, including `coord.npy`, `color.npy`, `opacity.npy`, `scale.npy`, `quat.npy`, `normal.npy`, `instance.npy`, and `segment*.npy` where available. For Chorus, add the compact teacher-feature files in the same split scene folders before chunking:

```text
lang_feat.npy
lang_feat_index.npy
dino_feat.npy
dino_feat_index.npy
pe_feat.npy
pe_feat_index.npy
```

The chunking script derives `valid_feat_mask.npy` from `lang_feat_index.npy` and writes it into the chunk folders.

During experiments, `pe_feat` is only used for ScanNet++ in the main Chorus multi-teacher setup.

The dataset split files are available at [data_splits](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k/tree/main/data_splits).

The split usage is:

- ScanNet: train on `train` and `test`, evaluate on `val`.
- ScanNet++: train on `nvs_sem_train` and `sem_test`, evaluate on `nvs_sem_val`.
- Matterport3D: prepare teacher features on scene-level `scenes_train` and `scenes_val`; evaluate on region-level `regions_test` through the released `test_eval` folder.

## Stage 1: 2D SigLIP2/SAM2 Features

Use [preprocess_2d_language_feature](https://github.com/unique1i/chorus_data_generator/tree/main/preprocess_2d_language_feature) to create one compact 2D archive per scene:

```bash
micromamba activate lang_feat
cd /path/to/chorus_data_generator/preprocess_2d_language_feature

python process_dataset.py \
  --txt_file /path/to/scannet_scenes.txt \
  --dataset_folder /path/to/scannet \
  --dataset_name scannet \
  --save_folder /path/to/language_features_siglip2_so400m \
  --scannet_image_subdir color_interval \
  --sam2_ckpt_path /path/to/sam2.1_hiera_large.pt \
  --compact_scene_outputs \
  --skip_existing_outputs
```

The output is:

```text
<language_feature_root>/<scene_id>/scene_outputs.npz
```

Run the same script with `--dataset_name scannetpp` or `--dataset_name matterport3d` for the other datasets. For Matterport3D, pass the OpenScene `matterport_2d` root as `--dataset_folder` and use the scene-level split files, such as `scenes_train.txt` and `scenes_val.txt`. The repo README includes setup notes for SAM2 checkpoints, flash-attn, and one-scene checks.

## Stage 2: Lift SigLIP2 To 3DGS

Use [occamlgs_dev](https://github.com/unique1i/chorus_data_generator/tree/main/occamlgs_dev) to lift the compact 2D language features onto the released 3D Gaussian rows:

```bash
micromamba activate occamlgs
cd /path/to/chorus_data_generator/occamlgs_dev

python batch_feature_extractor.py \
  --txt_file /path/to/scannet_train_scenes.txt \
  --dataset_folder /path/to/scannet \
  --gs_folder /path/to/scannet_mcmc_3dgs_lang_large \
  --dataset_name scannet \
  --lang_feat_path /path/to/language_features_siglip2_so400m \
  --language_features_name language_features_siglip2_so400m \
  --save_path /path/to/scannet_mcmc_3dgs_lang_large/train \
  --resolution -1 \
  --load_compact_npz \
  --load_gs_from_npy \
  --save_compact_feat \
  --output_format npy \
  --skip_existing_outputs
```

OccamLGS writes:

```text
<final_root>/<split>/<scene_id>/lang_feat.npy
<final_root>/<split>/<scene_id>/lang_feat_index.npy
```

Run this once per split, changing both `--txt_file` and `--save_path` to the matching split folder, such as `<final_root>/val` or `<final_root>/test`.

## Stage 3: Lift DINOv3 And PE-Spatial

Use [ludvig_dev](https://github.com/unique1i/chorus_data_generator/tree/main/ludvig_dev) for the visual teachers. DINOv3 is used for ScanNet, ScanNet++, and Matterport3D:

```bash
micromamba activate ludvig
cd /path/to/chorus_data_generator/ludvig_dev

python run.py \
  --split /path/to/scannet_train_scenes.txt \
  --dataset scannet \
  --colmap_root /path/to/scannet/scans \
  --gs_root /path/to/scannet_mcmc_3dgs_lang_large \
  --dst_dir /path/to/ludvig_dinov3_sidecars \
  --height 968 \
  --width 1296 \
  --models dinov3 \
  --load_gs_from_npy
```

For the Chorus release, DINOv3 uses the high-resolution image path for ScanNet and ScanNet++: ScanNet runs at `1296 x 968`, and ScanNet++ DSLR views run at `1752 x 1168`. PE-Spatial is different: use the regular selected-view JSON path, not the separate high-resolution DINO path.

For ScanNet++ PE-Spatial:

```bash
export PERCEPTION_MODELS_ROOT=/path/to/perception_models

python run.py \
  --split /path/to/scannetpp_train_scenes.txt \
  --dataset scannetpp \
  --colmap_root /path/to/scannetpp_v2/data \
  --gs_root /path/to/scannetpp_v2_mcmc_3dgs_lang_large \
  --dst_dir /path/to/ludvig_pe_spatial_sidecars \
  --height 1168 \
  --width 1752 \
  --models pe_spatial \
  --load_gs_from_npy
```

The Chorus output names are fixed:

```text
<ludvig_sidecar_root>/<scene_id>/dinov3/dino_feat.npy
<ludvig_sidecar_root>/<scene_id>/dinov3/dino_feat_index.npy

<ludvig_sidecar_root>/<scene_id>/pe_spatial/pe_feat.npy
<ludvig_sidecar_root>/<scene_id>/pe_spatial/pe_feat_index.npy
```

LUDVIG keeps the model name in the output path, so place the compact sidecars into the final split scene folders before chunking:

```bash
FINAL_ROOT=/path/to/scannet_mcmc_3dgs_lang_large
SPLIT=train
SCENES=/path/to/scannet_train_scenes.txt
DINO_ROOT=/path/to/ludvig_dinov3_sidecars
PE_ROOT=/path/to/ludvig_pe_spatial_sidecars  # optional

while read -r scene; do
  [ -z "$scene" ] && continue
  cp "$DINO_ROOT/$scene/dinov3/dino_feat.npy" "$FINAL_ROOT/$SPLIT/$scene/"
  cp "$DINO_ROOT/$scene/dinov3/dino_feat_index.npy" "$FINAL_ROOT/$SPLIT/$scene/"
  if [ -f "$PE_ROOT/$scene/pe_spatial/pe_feat.npy" ]; then
    cp "$PE_ROOT/$scene/pe_spatial/pe_feat.npy" "$FINAL_ROOT/$SPLIT/$scene/"
    cp "$PE_ROOT/$scene/pe_spatial/pe_feat_index.npy" "$FINAL_ROOT/$SPLIT/$scene/"
  fi
done < "$SCENES"
```

Repeat the placement for each split you processed. ScanNet and Matterport3D can use PE-Spatial sidecars if you generated them; the main Chorus release recipe uses PE-Spatial for ScanNet++.

Before chunking, each final scene folder should contain the downloaded base arrays plus the Chorus teacher sidecars:

```text
<final_root>/<split>/<scene_id>/coord.npy
<final_root>/<split>/<scene_id>/lang_feat.npy
<final_root>/<split>/<scene_id>/lang_feat_index.npy
<final_root>/<split>/<scene_id>/dino_feat.npy
<final_root>/<split>/<scene_id>/dino_feat_index.npy
<final_root>/<split>/<scene_id>/pe_feat.npy  # optional
<final_root>/<split>/<scene_id>/pe_feat_index.npy  # optional
```

Full LUDVIG runs can take a while on large scenes. For a quick one-scene check, add `--max-views 10`.

## Stage 4: Chunk Per-Scene Data

Chorus pretraining reads chunked folders. The chunking script [sampling_chunking_data_gs.py](pointcept/datasets/preprocessing/sampling_chunking_data_gs.py) preserves compact language, DINOv3, and PE-Spatial sidecars.

ScanNet and ScanNet++ use XY chunks:

```bash
python pointcept/datasets/preprocessing/sampling_chunking_data_gs.py \
  --dataset_root /path/to/scannet_mcmc_3dgs_lang_large \
  --split train \
  --grid_size 0.01 \
  --chunk_range 6 6 \
  --chunk_stride 4 4 \
  --chunk_minimum_size 50000 \
  --num_workers 8
```

Run the same command for the training/evaluation split folders you need. For example, ScanNet and ScanNet++ training reproduction should chunk both `train` and `test`; evaluation uses `val`.

Matterport3D uses XYZ chunks:

```bash
python pointcept/datasets/preprocessing/sampling_chunking_data_gs.py \
  --dataset_root /path/to/matterport3d_scene_mcmc_3dgs_lang_large \
  --split train \
  --grid_size 0.01 \
  --chunk_range 6 6 4 \
  --chunk_stride 4 4 4 \
  --chunk_z \
  --chunk_minimum_size 50000 \
  --num_workers 8
```

For Matterport3D, prepare chunks for scene-level `train` and `val` as training shards. These commands produce `train_grid1.0cm_chunk6x6x4_stride4x4x4` and `val_grid1.0cm_chunk6x6x4_stride4x4x4`. Evaluation stays on the released region-level `test_eval` folder, using `regions_test.txt`.

## Quick Checks

Before launching all scenes, run one-scene checks:

- 2D repo: confirm `<save_folder>/scene0000_00/scene_outputs.npz`.
- OccamLGS: confirm `<final_root>/<split>/<scene>/lang_feat.npy` and `lang_feat_index.npy`.
- LUDVIG DINOv3: confirm `<ludvig_sidecar_root>/<scene>/dinov3/dino_feat.npy` and `dino_feat_index.npy`.
- LUDVIG PE-Spatial: confirm `<ludvig_sidecar_root>/<scene>/pe_spatial/pe_feat.npy` and `pe_feat_index.npy`.
- Final placement: confirm a final split scene has base arrays such as `coord.npy` and `normal.npy`, plus compact teacher sidecars.
- Chunking: confirm a chunk folder has matching `*_feat.npy`, `*_feat_index.npy`, and `valid_feat_mask.npy` files.

# Additional Evaluation Data

## InteriorGS

We provide processed InteriorGS data for additional evaluation in our [interior_gs_preprocessed](https://huggingface.co/datasets/SceneSplatPro/interior_gs_preprocessed) data repo. Please request access there before downloading the files. In the Chorus paper, we use the InteriorGS `test` split as the benchmark split. The split files are provided in the same repo at  `metadata/splits`.

The starting point is the original [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) release. We converted the released 3DGS scenes into the similar per-scene `*.npy` files used by the rest of this repo.

After converting the 3DGS files, we also processed the original 3D bounding-box annotations. For each scene, we assign semantic labels to `segment.npy` and instance labels to `instance.npy` for the Gaussian rows, which are all 0-indexed and with the ignore label as -1. The assignment uses connected components, which helps keep the labels spatially consistent.

For semantic evaluation, we map the original 715 classes in the InteriorGS semantic taxonomy into our 72-class benchmark taxonomy based on the semantic hierarchy. The full mapping is included in the processed data repo as `metadata/semantic_mapping.csv`, and the 72-class names are in `metadata/taxonomy_labels.txt`. The index in the `segment.npy` corresponds to the row index in the class names.
