#!/bin/bash

dataset=$1
model=$2

python -m src.link_prediction.tune  --dataset $dataset --model $model
python -m src.link_prediction.train --dataset $dataset --model $model --valid 5
python -m src.link_prediction.test  --dataset $dataset --model $model
