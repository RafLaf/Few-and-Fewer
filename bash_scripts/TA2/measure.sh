#!/bin/bash
list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current task ID from the SLURM_ARRAY_TASK_ID environment variable
task_id=$SLURM_ARRAY_TASK_ID
den=11
# Get the current string from the list based on the task ID
dat=${list1[$((task_id / den))]}
index_subset=$((task_id % den))
clustering=$1
mode=$2
load_backbone_base="${WORK}/results/TA3/backbones/${clustering}/"
load_backbone="${load_backbone_base}/backbones_${index_subset}"
save_feat="${WORK}/results/TA3/features/${clustering}/${dat}/f_${index_subset}"
few_shot_runs="600"
dataset_path="${SCRATCH}/"
test_dataset="metadataset_${dat}_test"
epoch="1"
backbone="resnet12"
batch_size="128"
few_shot="--few-shot"

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
save_result="${WORK}/results/TA1em3/result_TA1em3_${mode}_${clustering}.pt"

python ../../../main.py \
    --freeze-backbone \
    --dataset-path "${dataset_path}" \
    --load-backbone "${load_backbone}" \
    --test-dataset "${test_dataset}" \
    --epoch "${epoch}" \
    --backbone "${backbone}" \
    --batch-size "${batch_size}" \
    --few-shot-shots "${few_shot_shots}" \
    --few-shot-ways "${few_shot_ways}" \
    --few-shot-queries "${few_shot_queries}" \
     ${few_shot} \
    --save-features-prefix "${save_feat}" \
    --save-test $save_result \
    --few-shot-runs $few_shot_runs

