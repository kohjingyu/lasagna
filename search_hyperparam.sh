#!/bin/bash

gpu=4

declare -a posw=(3 5 10 20 50 100)

for pos in "${posw[@]}"
do
CUDA_VISIBLE_DEVICES=$gpu python3 train.py --pos_weight $pos &> "logs/output_posw${pos}.txt"
done
