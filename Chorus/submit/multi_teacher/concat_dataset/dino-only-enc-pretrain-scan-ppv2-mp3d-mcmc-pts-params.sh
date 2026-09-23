#!/bin/bash
# This concat_dataset submit script uses the NCCL multi-node training template.
#SBATCH --output=logs/%A.log
#SBATCH --error=logs/%A.log
#SBATCH -p gpu_h100
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00

source ~/.bashrc
micromamba activate scene_splat

cd /home/yli7/projects/scene_3dgs_pro/chorus_release

echo "Running on $(hostname) | $(date)"
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR="${MASTER_NODE}.local.snellius.surf.nl"
MASTER_PORT=29501
echo "Master node: $MASTER_NODE"
echo "Master address: $MASTER_ADDR"
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,COLL
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME="eno"
export NCCL_SOCKET_TIMEOUT=7200
export NCCL_TIMEOUT=7200
export TORCH_NCCL_HEARTBEAT_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_WATCHDOG_TIMEOUT=7200

gpu_num=$(($SLURM_GPUS_PER_NODE*$SLURM_NNODES))
batch_size=$((1*gpu_num))
batch_size_val=$((1*gpu_num))
batch_size_test=$((1*gpu_num))
num_worker=$((6*gpu_num))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional resume training from a checkpoint:
#   resume=True \
#   weight=exp_runs/lang_pretrainer/multi_teacher/concat_dataset/dino-enc-pretrain-scan-ppv2-matt-mcmc-pts-params-submit/model/model_last.pth \
# Optional wandb settings (wandb is enabled by default):
#   enable_wandb=False \
#   wandb_id=<wandb_run_id>

srun python -u -m tools.train \
  --config-file configs/chorus/concat_dataset/dino-only-enc-pretrain-scan-ppv2-mp3d-mcmc-pts-params.py \
  --options \
    save_path=exp_runs/lang_pretrainer/chorus/concat_dataset/dino-only-enc-pretrain-scan-ppv2-mp3d-mcmc-pts-params \
    batch_size=$batch_size batch_size_val=$batch_size_val \
    batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
  --num-gpus $gpu_num \
  --multi_node
