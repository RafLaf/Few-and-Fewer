#!/bin/bash
mode=$1
dat_ind=$SLURM_ARRAY_TASK_ID


list_dat=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list_dat[$dat_ind]}

cd ../../../

dataset_path="${SCRATCH}/"
path_to_subsets=""${WORK}/episode_600/binary_agnostic_{}.npy""
task_file="${WORK}/episode_600/magnitude600_${mode}_test_${dat}.pt" 


module purge
module load pytorch-gpu/py3/1.12.1



python fim_dist_episodes.py \
    --dataset-path ${dataset_path} \
    --load-episodes "${WORK}/episode_600/magnitude600_${mode}_test_${dat}.pt" \
    --out-file "fim/result_${mode}_${dat}.pt" \
    --target-dataset metadataset_${dat}
