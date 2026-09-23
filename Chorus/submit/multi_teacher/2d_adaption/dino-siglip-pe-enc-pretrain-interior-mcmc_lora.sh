#!/bin/bash
#SBATCH --job-name=chorus_2d_dino_siglip_pe
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=240G
#SBATCH --time=7-0
#SBATCH --output=sbatch_log/chorus_2d_dino_siglip_pe.log

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
BASE_CKPT=${BASE_CKPT:?Set BASE_CKPT to the released Chorus checkpoint}
DATA_ROOT=${DATA_ROOT:-data/interior_gs_2d_adaptation}
SAVE_PATH=${SAVE_PATH:-exp_runs/lang_pretrainer/chorus/2d_adaption/dino-siglip-pe-enc-pretrain-interior-mcmc_lora}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_GPUS=${NUM_GPUS:-1}

cd "$REPO_ROOT"

"$PYTHON_BIN" -m tools.train \
  --config-file configs/chorus/2d_adaption/dino-siglip-pe-enc-pretrain-interior-mcmc_lora.py \
  --options \
    batch_size="$BATCH_SIZE" \
    save_path="$SAVE_PATH" \
    model.backbone_path="$BASE_CKPT" \
    data.train.data_root="$DATA_ROOT" \
    data.val.data_root="$DATA_ROOT" \
  --num-gpus "$NUM_GPUS"
