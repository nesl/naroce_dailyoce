#!/bin/bash

# config="configs/naroce_finetune_train.json"
config="configs/naroce_finetune_train_realistic_w2.0s.json"
adapter_model=mamba1_6L
nar_dataset=40000

for sensor_dataset in 2000; do
    for seed in 0 17 1243 3674 7341 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719; do
        echo "Finetuning NAROCE: config=$config, adapter=$adapter_model, nar_dataset=$nar_dataset, sensor_dataset=$sensor_dataset, seed=$seed"
        CUDA_VISIBLE_DEVICES=0 python naroce.py $config $nar_dataset $seed --sensor_dataset $sensor_dataset --adapter_model $adapter_model
    done
done
