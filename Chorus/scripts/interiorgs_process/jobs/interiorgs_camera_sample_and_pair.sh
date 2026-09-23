#!/bin/bash
set -euo pipefail

RAW_SCENE_ROOT=${RAW_SCENE_ROOT:?Set RAW_SCENE_ROOT to the InteriorGS raw scene root}
PREPROCESSED_ROOT=${PREPROCESSED_ROOT:?Set PREPROCESSED_ROOT to the preprocessed GS root}
SPLIT=${SPLIT:-train}
PAIR_TOP_K=${PAIR_TOP_K:-4}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SCRIPT_DIR"

python sample_camera_view_and_create_transform_interiorgs.py \
  --interior_gs_path "$RAW_SCENE_ROOT" \
  --interior_gs_preprocessed_path "$PREPROCESSED_ROOT"

python augment_the_visable_points_bbox_interiorgs.py \
  --scene_root_path "$PREPROCESSED_ROOT/$SPLIT"

python pair_the_visable_frames_bbox_interiorgs.py \
  --scene_root_path "$PREPROCESSED_ROOT/$SPLIT" \
  --batch_size "$PAIR_TOP_K"
