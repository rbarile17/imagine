#!/bin/bash

dataset=$1
model=$2
mode=$3
method=$4
summarization=$5

python -m src.explain                        --dataset $dataset --model $model --method $method --mode $mode --summarization $summarization
python -m src.evaluation.verify_explanations --dataset $dataset --model $model --method $method --mode $mode --summarization $summarization
python -m src.evaluation.compute_metrics     --dataset $dataset --model $model --method $method --mode $mode --summarization $summarization
