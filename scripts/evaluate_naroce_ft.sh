#!/bin/bash

# config="configs/ft_naroce_eval.json"
config="configs/ft_naroce_eval_realistic_w2.0s.json"

model=naroce_mamba1_6L 
nar_dataset=40000
# eval_dataset='5min'

for eval_dataset in '15min' '30min' ; do #'3min' '15min' '30min' '5min'
    for sensor_dataset in 2000; do
        for seed in 0 17 1243 3674 7341 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
            echo "Evaluating NAROCE: config=$config, model=$model, nar_dataset=$nar_dataset, sensor_dataset=$sensor_dataset, eval_dataset=$eval_dataset, seed=$seed"
            CUDA_VISIBLE_DEVICES=0 python evaluate.py $config $model $sensor_dataset $eval_dataset $seed --nar_dataset $nar_dataset
        done
    done
done

