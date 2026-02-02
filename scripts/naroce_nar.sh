#!/bin/bash

config="configs/naroce_nar_train.json"
nar_dataset=40000

for seed in 97 103; do # 53 97 103 191 99719; do
    echo "Training NAR: config=$config, nar_dataset=$nar_dataset, seed=$seed"
    CUDA_VISIBLE_DEVICES=0 python naroce.py $config $nar_dataset $seed
done