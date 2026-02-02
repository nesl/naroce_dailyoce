#!/bin/bash

config="configs/baseline_eval_shift.json"
model=lstm

for shift in 0.05 0.1 0.15 0.2; do
    for eval_dataset in '5min' '15min' '30min' ; do # '15min' '30min' '5min'
        for sensor_dataset in 10000; do # 2000 4000 6000 8000 10000
            for seed in 0 17 1243 3674 7341 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
                echo "Evaluating Baseline: config=$config, model=$model, sensor_dataset=$sensor_dataset, eval_dataset=$eval_dataset, seed=$seed"
                python evaluate.py $config $model $sensor_dataset $eval_dataset $seed --shift $shift
            done
        done
    done
done