#!/bin/bash
mode=$1
dat_ind=$2
list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list1[$dat_ind]}
mag_or_ncm="magnitude"
load_backbone_base="${WORK}/results/TI2/${mode}/backbones/${dat}"
load_backbone="${load_backbone_base}/backbones_${SLURM_ARRAY_TASK_ID}"
loadepisodes="${WORK}/episode_600/${mag_or_ncm}600_${mode}_test_${dat}.pt"
indexepisode=$SLURM_ARRAY_TASK_ID
dataset_path="${SCRATCH}/"
test_dataset="metadataset_${dat}_test"
epoch="1"
few_shot_runs="1"
index_subset=$SLURM_ARRAY_TASK_ID
backbone="resnet12"
batch_size="128"
few_shot="--few-shot"
lr=0.001

if [ "$mode" == "MD" ]; then
    few_shot_shots="0"
    few_shot_ways="0"
    few_shot_queries="0"
elif [ "$mode" == "1s5w" ]; then
    few_shot_shots="1"
    few_shot_ways="5"
    few_shot_queries="15"
elif [ "$mode" == "5s5w" ]; then
    few_shot_shots="5"
    few_shot_ways="5"
    few_shot_queries="15"
else
    echo "Invalid mode. Please choose between MD, 1s5w, and 5s5w."
    exit 1
fi
save_stats="${WORK}/results/TI2/measure/${mode}/${dat}/${SLURM_ARRAY_TASK_ID}.json"

python ../../../main.py \
    --freeze-backbone \
    --dataset-path "${dataset_path}" \
    --load-backbone "${load_backbone}" \
    --load-episodes "${loadepisodes}" \
    --test-dataset "${test_dataset}" \
    --index-episode "${indexepisode}" \
    --epoch "${epoch}" \
    --backbone "${backbone}" \
    --batch-size "${batch_size}" \
    --few-shot-shots "${few_shot_shots}" \
    --few-shot-ways "${few_shot_ways}" \
    --few-shot-queries "${few_shot_queries}" \
    --few-shot-runs "${few_shot_runs}" \
    ${few_shot} \
    ${task_queries} \
    --save-stats $save_stats \

