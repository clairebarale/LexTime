#!/usr/bin/env bash


export CACHE_DIR="/.cache"
export ACCESS_TOKEN="access_token"

python code/inference/temporal_ordering.py \
    --prompt_type="base" \
    --model_name="meta-llama/Meta-Llama-3.1-8B" \
    --cache_dir=$CACHE_DIR \
    --access_token=$ACCESS_TOKEN \
    --input_file="data/splits/lextime_512samples.csv" \
    --output_file="output.csv"  
