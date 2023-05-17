#!/bin/bash
cd ../../../
dataset_path="${SCRATCH}/"
path_to_subsets="${WORK}/episode_600/binary_agnostic_{}.npy"

module purge
module load pytorch-gpu/py3/1.12.1


set -x
python fim_distTA.py \
    --dataset-path "${dataset_path}" \
    --info ${path_to_subsets}
