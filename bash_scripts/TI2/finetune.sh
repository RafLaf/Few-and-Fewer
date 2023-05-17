#!/bin/bash
mode=$1
dat_ind=$2

list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list1[$dat_ind]}

dataset_path="${SCRATCH}/"
load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
subset_file="${WORK}/episode_600/binaryFS600_test_${mode}_50_${dat}.npy"
index_subset=$SLURM_ARRAY_TASK_ID
training_dataset="metadataset_imagenet_train"
epoch="10"
dataset_size="10000"
wd="0.0001"
scheduler="cosine"
backbone="resnet12"
batch_size="128"
few_shot_shots="0"
few_shot_ways="0"
few_shot_queries="0"
few_shot="--few-shot"
lr=0.001

save_backbone_base="${WORK}/results/TI2/${mode}/backbones/${dat}/"
save_classifier_base="${WORK}/results/TI2/${mode}/classifiers/${dat}/"

save_backbone="${save_backbone_base}/backbones_${index_subset}"
save_classifier="${save_classifier_base}/classifier_finetune_${index_subset}"
load_classifier="${save_classifier_base}/classifier_${index_subset}"



python ../../../main.py \
    --dataset-path "${dataset_path}" \
    --load-backbone "${load_backbone}" \
    --subset-file "${subset_file}" \
    --index-subset "${index_subset}" \
    --training-dataset "${training_dataset}" \
    --epoch "${epoch}" \
    --dataset-size "${dataset_size}" \
    --wd "${wd}" \
    --lr "${lr}" \
    --scheduler "${scheduler}" \
    --backbone "${backbone}" \
    --batch-size "${batch_size}" \
    --few-shot-shots "${few_shot_shots}" \
    --few-shot-ways "${few_shot_ways}" \
    --few-shot-queries "${few_shot_queries}" \
    ${few_shot} \
    --save-backbone "${save_backbone}" \
    --save-classifier "${save_classifier}" \
    --load-classifier "${load_classifier}"