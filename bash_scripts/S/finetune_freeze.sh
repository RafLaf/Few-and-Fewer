#!/bin/bash
mode=$1
dat_ind=$2


list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list1[$dat_ind]}
mag_or_ncm="magnitude"
dataset_path="${SCRATCH}/"
load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
task_file="${WORK}/episode_600/${mag_or_ncm}600_${mode}_test_${dat}.pt" 
index_subset=$SLURM_ARRAY_TASK_ID
train_dataset="metadataset_${dat}_test"
epoch="15"
wd="0.0002"
scheduler="cosine"
backbone="resnet12"
batch_size="128"
lr=0.01
optimizer="Adam"
freeze_backbone="--freeze-backbone --force-train"


save_classifier_base="${WORK}/results/S/${mode}/classifiers/${dat}/"
save_classifier="${save_classifier_base}/classifier_finetune_freeze_${SLURM_ARRAY_TASK_ID}"


python ../../../main.py \
    --dataset-path "${dataset_path}" \
    --load-backbone "${load_backbone}" \
    --task-file "${task_file}" \
    --index-subset "${index_subset}" \
    --training-dataset "${train_dataset}" \
    --epoch "${epoch}" \
    --wd "${wd}" \
    --lr "${lr}" \
    --scheduler "${scheduler}" \
    --backbone "${backbone}" \
    --batch-size "${batch_size}" \
    --save-classifier "${save_classifier}" \
    --optimizer ${optimizer} \
    ${freeze_backbone}