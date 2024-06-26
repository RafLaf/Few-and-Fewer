#!/bin/bash
list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list1[$SLURM_ARRAY_TASK_ID]}

dataset_path="${SCRATCH}/"
load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
subset_file="${WORK}/episode_600/binary_${dat}_50.npy"
index_subset="0"
training_dataset="metadataset_imagenet_train"
epoch="20"
dataset_size="10000"
wd="0.0001"
scheduler="cosine"
backbone="resnet12"
batch_size="128"
few_shot_shots="0"
few_shot_ways="0"
few_shot_queries="0"
few_shot="--few-shot"
lr=0.005

save_backbone_base="${WORK}/results/DI/backbones/${dat}/"
save_classifier_base="${WORK}/results/DI/classifiers/${dat}/"
save_backbone="${save_backbone_base}/backbones_${epoch}_${lr}"
save_classifier="${save_classifier_base}/classifier_finetune"



module purge # nettoyer les modules herites par defaut
#conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.12.1 # charger les modules
set -x # activer l’echo des commandes

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
    --save-classifier "${save_classifier}"