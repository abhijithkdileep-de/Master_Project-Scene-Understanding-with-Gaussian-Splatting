#!/bin/bash
#SBATCH --output=logs/%A.log
#SBATCH --error=logs/%A.log
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00

source ~/.bashrc
micromamba activate scene_splat

cd /home/yli7/projects/scene_3dgs_pro/chorus_release

echo "Running on $(hostname) | $(date)"

gpu_num=$(($SLURM_GPUS_PER_NODE*$SLURM_NNODES))
batch_size=$((1*gpu_num))
batch_size_val=$((1*gpu_num))
batch_size_test=$((1*gpu_num))
num_worker=$((6*gpu_num))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional resume training from a checkpoint:
#   resume=True \
#   weight=<checkpoint_path> \
# Optional wandb settings (wandb is enabled by default):
#   enable_wandb=False \
#   wandb_id=<wandb_run_id>

python -u -m tools.train \
  --config-file configs/chorus/lang-only-enc-pretrain-mp3d-mcmc.py \
  --options \
    save_path=exp_runs/lang_pretrainer/chorus/lang-only-enc-pretrain-mp3d-mcmc \
    batch_size=$batch_size batch_size_val=$batch_size_val \
    batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
  --num-gpus $gpu_num
