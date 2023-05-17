#!/bin/bash
mode=$1
dat_ind=$2

BATCH_SIZE=100

start=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
end=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE))
set -x # activer l’echo des commandes
# Loop through the task IDs in the specified range
echo $start
echo $end
list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
set -x # activer l’echo des commandes
for task_id in $(seq $start $((end - 1))); do
# Get the current string from the list based on the task ID
    dat=${list1[$dat_ind]}
    mag_or_ncm="magnitude"
    dataset_path="${SCRATCH}/"
    load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
    task_file="${WORK}/episode_600/${mag_or_ncm}600_${mode}_test_${dat}.pt" \
    index_episode=$task_id
    train_dataset="metadataset_${dat}_test"
    epoch="10"
    wd="0.0002"
    scheduler="cosine"
    backbone="resnet12"
    batch_size="128"
    lr=0.0001
    optimizer="adam"

    save_backbone_base="${WORK}/results/S/${mode}/backbones/${dat}/"
    save_backbone="${save_backbone_base}/backbones1e4_${task_id}"
    save_classifier_base="${WORK}/results/S/${mode}/classifiers/${dat}/"
    save_classifier="${save_classifier_base}/classifier_finetune1e4_${task_id}"



    python ../../../main.py \
        --dataset-path "${dataset_path}" \
        --load-backbone "${load_backbone}" \
        --task-file "${task_file}" \
        --index-episode "${index_episode}" \
        --training-dataset "${train_dataset}" \
        --epoch "${epoch}" \
        --wd "${wd}" \
        --lr "${lr}" \
        --scheduler "${scheduler}" \
        --backbone "${backbone}" \
        --batch-size "${batch_size}" \
        --save-backbone "${save_backbone}" \
        --save-classifier "${save_classifier}" \
        --optimizer ${optimizer}
done