#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod

method=kelpie
mode=ri
dataset=DB50K
summarization=no
entity_density=any
pred_rank=notfirst

./scripts/explain.sh $dataset ConvE $mode $method $summarization $entity_density $pred_rank > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}_${entity_density}_${pred_rank}.log" 2>&1
