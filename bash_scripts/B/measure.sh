#!/bin/bash
load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
save_feat="${WORK}/results/B/f_baseline"
few_shot_runs="600"
dataset_path="${SCRATCH}/"
test_dataset="metadataset_${DATASET_OPTION}_test"
epoch="1"
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
save_result="${WORK}/results/B/result_baseline_${mode}.pt"

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

