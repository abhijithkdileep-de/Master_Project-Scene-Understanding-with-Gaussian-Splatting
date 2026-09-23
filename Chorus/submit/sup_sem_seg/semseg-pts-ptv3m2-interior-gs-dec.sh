#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00

# conda activation
source ~/.bashrc
micromamba activate scene_splat

cd /home/yli7/projects/scene_3dgs_pro/chorus_release

echo "Running on $(hostname) | $(date)"
gpu_num=$(($SLURM_GPUS_PER_NODE*$SLURM_NNODES))
batch_size=$((3*gpu_num))
batch_size_val=$((1*gpu_num))
batch_size_test=$((1*gpu_num))
num_worker=$((8*gpu_num))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m tools.train \
  --config-file configs/interior_gs/semseg-pts-ptv3m2-interior-gs-dec.py \
  --options \
    save_path=exp_runs/sup_sem_seg/semseg-pts-sonata-interior-gs-dec-probe \
    batch_size=$batch_size batch_size_val=$batch_size_val \
    batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num wandb_project=pointcept_base_exp \
    weight=exp_runs/ckpt/pretrain-sonata-v1m1-0-base.pth \
  --num-gpus $gpu_num \
