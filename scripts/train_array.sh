#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-6


config=./scripts/datasets.txt

dataset=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

./scripts/train.sh $dataset TransE  > "./logs/train/TransE_${dataset}.log"  2>&1 & 
./scripts/train.sh $dataset ConvE   > "./logs/train/ConvE_${dataset}.log"   2>&1 &
./scripts/train.sh $dataset ComplEx > "./logs/train/ComplEx_${dataset}.log" 2>&1 &
wait
