#!/bin/bash
list_dat=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
list_sampling=("1s5w" "5s5w" "MD")
# Get the current string from the list based on the task ID

cd ../../../

dataset_path="${SCRATCH}/"
path_to_subsets=""${WORK}/episode_600/binary_agnostic_{}.npy""
task_file="${WORK}/episode_600/magnitude600_${mode}_test_${dat}.pt" 
valtest="test"
few_shot_run=600
subset_dir="${WORK}/episode_600/"
load_logits="${WORK}/logits/logits_{}_{}.pt"

module purge
module load pytorch-gpu/py3/1.12.1


for dat in "${list_dat[@]}"; do
  for mode in "${list_sampling[@]}"; do

        python AA_heuristic.py \
            --dataset-path ${dataset_path} \
            --load-episodes "${WORK}/episode_600/magnitude600_${mode}_test_${dat}.pt" \
            --out-file "AA4TA/AA4TA_${mode}_${dat}.npy" \
            --target-dataset ${dat} \
            --valtest ${valtest} \
            --few-shot-run 600 \
            --subset-file ${subset_dir} \
            --load-logits "${load_logits}" 
    done
done
