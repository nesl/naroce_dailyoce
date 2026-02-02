#!/bin/bash

config="configs/baseline_train.json"
# config="configs/baseline_train_realistic_w2.0s.json"
model=lstm

for dataset in 6000 8000; do # 2000 4000 6000 8000 10000
    for seed in 0 17 1243 3674 7341 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
        echo "Training: config=$config, model=$model, dataset=$dataset, seed=$seed"
        python baseline_debug.py $config $model $dataset $seed
    done
done
