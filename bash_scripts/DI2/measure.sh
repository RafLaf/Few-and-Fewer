#!/bin/bash
list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list1[$SLURM_ARRAY_TASK_ID]}

load_backbone_base="${WORK}/results/DI/backbones/${dat}"
load_backbone="${load_backbone_base}/backbones_20_0.01"
load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
save_feat="${WORK}/results/DI/features/${dat}/f_baseline"
few_shot_runs="600"
dataset_path="${SCRATCH}/"
test_dataset="metadataset_${dat}_test"
epoch="1"
backbone="resnet12"
batch_size="128"
few_shot="--few-shot"
lr=0.001
mode=$1

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
save_result="${WORK}/results/DI/result_DI_0.0001_${mode}.pt"

module purge # nettoyer les modules herites par defaut
#conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.12.1 # charger les modules
set -x # activer l’echo des commandes

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
    --few-shot ${few_shot} \
    --save-features-prefix "${save_feat}" \
    --save-test $save_result \
    --few-shot-runs $few_shot_runs

