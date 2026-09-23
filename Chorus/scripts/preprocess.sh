#!/bin/bash
#SBATCH --job-name=submit  # Job name
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p capacity      # performance for RTX 6000 Ada 48G, use 'capacity' for 24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1   # 1 for single gpu
#SBATCH --mem=120G           # max 90G for single gpu
#SBATCH --time=24:00:00     # hour
##SBATCH --nodelist=hipster-cn006

source /home/yli7/.bashrc
micromamba activate scene_splat

cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro
echo "Job Start at $(date)"
echo "running on $(hostname)"

############## preprocess chunking data   ##################


############## preprocess holicity_mcmc_3dgs version ##################
# cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing/holicity
# python -u preprocess_holicity.py \
#     --input_root /gpfs/work3/0/prjs1291/datasets/holicity/perspective/collected_by_region \
#     --output_root /gpfs/work3/0/prjs1291/datasets/ptv3_preprocessed/holicity

# cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing/holicity
# python -u preprocess_holicity_gs.py \
#     --pc_root  /gpfs/work3/0/prjs1291/datasets/ptv3_preprocessed/holicity \
#     --gs_root  /gpfs/work3/0/prjs1291/datasets/gaussian_world/holicity_mcmc_3dgs \
#     --feat_root /home/yli7/scratch2/datasets/gaussian_world/holicity_mcmc_3dgs/language_features_siglip2 \
#     --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/holicity_mcmc_3dgs \
#     --split_folder /gpfs/work3/0/prjs1291/datasets/holicity/splits/ours \
#     --num_workers 12 \
#     --remove_feat \

# cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/holicity_mcmc_3dgs  \
#         --grid_size 0.005 --chunk_range 6 6 --chunk_stride 4 4 --split val --num_workers 12

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/holicity_mcmc_3dgs  \
#         --grid_size 0.005 --chunk_range 6 6 --chunk_stride 4 4 --split train --num_workers 12


# ############## preprocess matterport_scene_mcmc_3dgs version ##################
# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing/matterport3d
# python -u preprocess_matterport3d_gs.py \
#     --pc_root  /home/yli7/scratch2/datasets/ptv3_preprocessed/matterport3d_house \
#     --gs_root  /home/yli7/scratch2/datasets/gaussian_world/matterport3d_scene_mcmc_3dgs \
#     --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_scene_mcmc_3dgs_lang_large \
#     --feat_root /home/yli7/scratch2/datasets/gaussian_world/matterport3d_scene_mcmc_3dgs/language_features_siglip2_so400m \
#     --dino_root /home/yli7/scratch2/datasets/gaussian_world/matterport3d_scene_mcmc_3dgs/language_features_dinov3 \
#     --num_workers 4 \
#     # --remove_feat \
#     # --single_process \

# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_scene_mcmc_3dgs_lang_large  \
#         --output_dir /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_scene_mcmc_3dgs_lang_large \
#         --grid_size 0.01 --chunk_range 6 6 4 --chunk_stride 4 4 4 --chunk_z --split train --num_workers 4 \
#         --max_chunk_num 250 --chunk_minimum_size 10000

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_scene_mcmc_3dgs_lang_large  \
#         --output_dir /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_scene_mcmc_3dgs_lang_large \
#         --grid_size 0.01 --chunk_range 6 6 4 --chunk_stride 4 4 4 --chunk_z --split val --num_workers 4 \
#         --max_chunk_num 250 --chunk_minimum_size 10000

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_scene_mcmc_3dgs_lang_large  \
#         --output_dir /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_scene_mcmc_3dgs_lang_large \
#         --grid_size 0.01 --chunk_range 6 6 4 --chunk_stride 4 4 4 --chunk_z --split test --num_workers 4 \
#         --max_chunk_num 250 --chunk_minimum_size 10000

# ############## preprocess matterport_region_mcmc_3dgs version ##################
# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing/matterport3d
# python -u preprocess_matterport3d_gs.py \
#     --pc_root  /home/yli7/scratch2/datasets/ptv3_preprocessed/matterport3d \
#     --gs_root  /home/yli7/scratch2/datasets/gaussian_world/matterport3d_region_mcmc_3dgs \
#     --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs_lang_large \
#     --num_workers 6 \
#     --feat_root /home/yli7/scratch2/datasets/gaussian_world/matterport3d_region_mcmc_3dgs/language_features_siglip2_so400m \
#     --dino_root /home/yli7/scratch2/datasets/gaussian_world/matterport3d_region_mcmc_3dgs/language_features_dinov3 \
#     # --remove_feat \

# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs_lang_large  \
#         --output_dir /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs_lang_large \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 4 4 --split val --num_workers 6 --max_chunk_num 8 

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs_lang_large  \
#         --output_dir /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs_lang_large \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 4 4 --split test --num_workers 6

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs  \
#         --output_dir /home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs \
#         --subset_list /gpfs/work3/0/prjs1291/datasets/matterport3d/splits/splits_region/regions_train_subset.txt \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 2 --max_chunk_num 3
    
# ############## preprocess scannet_fix_xyz_gs version ##################
# cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing/scannet
# python -u preprocess_scannet_gs.py \
#     --dataset_root /home/yli7/scratch2/datasets/scannet \
#     --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_default_fix_xyz_gs \
#     --num_workers 12 \
#     --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannet_default_fix_xyz_gs \
#     --feat_root /home/yli7/scratch2/datasets/gaussian_world/scannet_default_fix_xyz_gs/language_features_siglip2 \
#     --feat_only \
#     # --skip_feat
#     # --feat_only \
#     # --pc_root /home/yli7/scratch2/datasets/ptv3_preprocessed/scannet_preprocessed \


# ############# preprocess scannet_mcmc_3dgs version ##################
# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing/scannet
# python -u preprocess_scannet_gs.py \
#     --dataset_root /home/yli7/scratch2/datasets/scannet \
#     --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs \
#     --num_workers 8 \
#     --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannet_mcmc_3dgs \
#     --feat_root /home/yli7/scratch2/datasets/gaussian_world/scannet_mcmc_3dgs/language_features_siglip2_so400m \
#     # --dino_root /home/yli7/scratch2/datasets/gaussian_world/scannet_mcmc_3dgs/language_features_dinov3 \
#     # --skip_feat \
#     # --feat_only \

# # temp, process the default version of 3dgs
# export PYTHONPATH=.
# python -u pointcept/datasets/preprocessing/scannet/preprocess_scannet_gs.py \
#     --dataset_root /home/yli7/scratch2/datasets/scannet \
#     --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_default_3dgs \
#     --num_workers 8 \
#     --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannet_default_3dgs \
#     --opacity_thre 0.01 \
#     --nn_dist_thre 0.25 \
#     --split val \
#     --skip_feat

# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 4 4 --split val --num_workers 8

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 4 4 --split test --num_workers 8

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 4 4 --split train --num_workers 8


############ preprocess scannetpp_v2_mcmc_3dgs version ################
# cd pointcept/datasets/preprocessing/scannetpp
# python -u preprocess_scannetpp_gs.py \
#     --dataset_root /home/yli7/scratch2/datasets/scannetpp_v2 \
#     --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_3dgs_new \
#     --num_workers 12 \
#     --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_mcmc_3dgs \
#     --pc_root /home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v2_preprocessed \
#     --feat_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_mcmc_3dgs/language_features_siglip2_so400m \
#     --dino_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_mcmc_3dgs/language_features_dinov3 \
#     # --scenes_list /home/yli7/projects/yue/GS_Transformer/temp.txt \
#     # --skip_feat
#     # --feat_only \

# temp, process the default version of 3dgs
export PYTHONPATH=.
python -u pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp_gs.py \
    --dataset_root /home/yli7/scratch2/datasets/scannetpp_v2 \
    --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_150k_3dgs \
    --num_workers 8 \
    --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_mcmc_150k_3dgs \
    --pc_root /home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v2_preprocessed \
    --scenes_list /home/yli7/scratch2/datasets/scannetpp_v2/splits/nvs_sem_val.txt \
    --opacity_thre -1 \
    --nn_dist_thre 0.10 \
    --skip_feat
    # --feat_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_mcmc_3dgs/language_features_siglip2_so400m \
    # --dino_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_mcmc_3dgs/language_features_dinov3 \
    # --feat_only \

# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_3dgs_new  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 4 4 --split train --num_workers 12

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_3dgs_new  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 4 4 --split test --num_workers 12

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_3dgs_new  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 4 4 --split val --num_workers 12


# cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers 12

############ preprocess scannetpp_v2_default_fix_xyz_gs version ################
# cd pointcept/datasets/preprocessing/scannetpp
# python -u preprocess_scannetpp_gs.py \
#     --dataset_root /home/yli7/scratch2/datasets/scannetpp_v2 \
#     --output_root /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs \
#     --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_default_fix_xyz_gs \
#     --pc_root /home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v2_preprocessed \
#     --dino_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_default_fix_xyz_gs/language_features_dinov3 \
#     --num_workers 6 \
#     --skip_feat
#     #     --feat_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_default_fix_xyz_gs/language_features_siglip2 \

# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers 8 --chunk_minimum_size 25000

# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split test --num_workers 8 --chunk_minimum_size 25000

# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 8 --chunk_minimum_size 25000


# ############ preprocess interior_gs version ################
# cd /home/yli7/projects/scene_3dgs_pro/scenesplat_pro/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/interior_gs  \
#         --grid_size 0.01 --chunk_range 8 8 --chunk_stride 6 6 --split train --num_workers 8 \
#         --chunk_minimum_size 50000


# ########### preprocess scannetpp_v1_default_fix_xyz_gs version ################
# python -u preprocess_scannetpp_gs_fixed_xyz.py \
#     --dataset_root /home/yli7/scratch2/datasets/scannetpp_v1 \
#     --output_root /gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs \
#     --num_workers 12 \
#     --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v1_default_fix_xyz_gs \
#     --pc_root /home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v1_preprocessed \
#     --feat_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v1_default_fix_xyz_gs/language_features_siglip2 \
#     --feat_only \

# cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 12

# cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split test --num_workers 12

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers 12

# ########### preprocess scannetpp_v1_3dgs_mcmc_depth_true version
# echo "Preprocess scannetpp_v1_3dgs_mcmc_depth_true version...\n"
# python -u preprocess_scannetpp_gs_feat.py \
#         --dataset_root /home/yli7/scratch2/datasets/scannetpp_v1 \
#         --output_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_3dgs_mcmc_depth_true \
#         --num_workers 16 \
#         --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v1_mcmc_3dgs \
#         --pc_root /home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v1_preprocessed \
#         --feat_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_v1_mcmc_3dgs/language_features \

# echo "Now chunking data....\n"
# cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_3dgs_mcmc_depth_true  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 10
        
# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_3dgs_mcmc_depth_true  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split test --num_workers 10

# python -u sampling_chunking_data_gs.py \
#         --dataset_root  /home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_3dgs_mcmc_depth_true  \
#         --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers 10

echo ""
echo "Job End!"
echo "Time: $(date)"