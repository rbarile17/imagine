#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-24

config=./scripts/config.txt

method=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $2}' $config)
mode=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $3}' $config)
dataset=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $4}' $config)
summarization=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $5}' $config)

if [[ $method = "criage" ]]; then
    ./scripts/explain.sh $dataset ConvE   $method $mode > "./logs/explain/${method}_ConvE_${dataset}_${mode}.log" 2>&1 &
    ./scripts/explain.sh $dataset ComplEx $method $mode > "./logs/explain/${method}_ComplEx_${dataset}_${mode}.log" 2>&1 &
fi
if [[ $method = "data_poisoning" ]]; then
    ./scripts/explain.sh $dataset ConvE   $method $mode > "./logs/explain/${method}_ConvE_${dataset}_${mode}.log" 2>&1 &
    ./scripts/explain.sh $dataset ComplEx $method $mode > "./logs/explain/${method}_ComplEx_${dataset}_${mode}.log" 2>&1 &
    ./scripts/explain.sh $dataset TransE  $method $mode > "./logs/explain/${method}_TransE_${dataset}_${mode}.log" 2>&1 & 
fi
if [[ $method = "imagine" ]]; then
    ./scripts/explain.sh $dataset ConvE   $method $mode $summarization > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}.log"   2>&1 &
    ./scripts/explain.sh $dataset ComplEx $method $mode $summarization > "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}.log" 2>&1 &
    ./scripts/explain.sh $dataset TransE  $method $mode $summarization > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}.log"  2>&1 & 
fi
if [[ $method = "i-kelpie" ]]; then
    ./scripts/explain.sh $dataset ConvE   $method $mode $summarization > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}.log"   2>&1 &
    ./scripts/explain.sh $dataset ComplEx $method $mode $summarization > "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}.log" 2>&1 &
    ./scripts/explain.sh $dataset TransE  $method $mode $summarization > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}.log"  2>&1 & 
fi
if [[ $method = "i-kelpie++" ]]; then
    ./scripts/explain.sh $dataset ConvE   $method $mode $summarization > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}.log"   2>&1 &
    ./scripts/explain.sh $dataset ComplEx $method $mode $summarization > "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}.log" 2>&1 &
    ./scripts/explain.sh $dataset TransE  $method $mode $summarization > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}.log"  2>&1 & 
fi
if [[ $method = "kelpie" ]]; then
    ./scripts/explain.sh $dataset ConvE   $method $mode                > "./logs/explain/${method}_ConvE_${dataset}_${mode}.log"                    2>&1 &
    ./scripts/explain.sh $dataset ComplEx $method $mode                > "./logs/explain/${method}_ComplEx_${dataset}_${mode}.log"                  2>&1 &
    ./scripts/explain.sh $dataset TransE  $method $mode                > "./logs/explain/${method}_TransE_${dataset}_${mode}.log"                   2>&1 & 
fi
if [[ $method = "kelpie++" ]]; then
    ./scripts/explain.sh $dataset ConvE   $method $mode $summarization > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}.log"   2>&1 &
    ./scripts/explain.sh $dataset ComplEx $method $mode $summarization > "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}.log" 2>&1 &
    ./scripts/explain.sh $dataset TransE  $method $mode $summarization > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}.log"  2>&1 & 
fi
if [[ $method = "w-imagine" ]]; then
    ./scripts/explain.sh $dataset ConvE   $method $mode $summarization > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}.log"   2>&1 &
    ./scripts/explain.sh $dataset ComplEx $method $mode $summarization > "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}.log" 2>&1 &
    ./scripts/explain.sh $dataset TransE  $method $mode $summarization > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}.log"  2>&1 & 
fi
wait
