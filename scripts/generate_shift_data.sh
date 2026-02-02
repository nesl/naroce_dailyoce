#!/bin/bash

cd data
for shift in 0.05 0.15; do
    python dataset_loader.py --duration 5 --split test --shift $shift

    python dataset_loader.py --duration 15 --split test --shift $shift

    python dataset_loader.py --duration 30 --split test --shift $shift
done

cd ..
