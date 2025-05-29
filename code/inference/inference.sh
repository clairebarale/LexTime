#!/usr/bin/env bash

export GOOGLE_GENAI_API_KEY="gemini_api_key"

PROMPT_TYPE="cot_few_shot"  # Options: base, one_shot, few_shot, cot_one_shot, cot_few_shot
INPUT_FILE="data/splits/lextime_512samples.csv" # Path to your input CSV
MODEL_NAME="gemini-2.5-pro-preview-05-06" # https://ai.google.dev/gemini-api/docs/models # 1000 requests per day maximum with my Tier1 key. 
#MODEL_NAME="gemini-2.5-flash-preview-04-17"
OUTPUT_FILE="results/${MODEL_NAME}/output_512_${PROMPT_TYPE}_geminipro3.csv"
# temporature at 0.5. max output tokens at 150


python3 code/inference/gemini_inference.py \
    --prompt_type=$PROMPT_TYPE \
    --model_name=$MODEL_NAME \
    --input_file=$INPUT_FILE \
    --output_file=$OUTPUT_FILE \


# uncomment if you want to use non-gemini models:

#export CACHE_DIR="/.cache"
#export ACCESS_TOKEN="access_token"

#python code/inference/temporal_ordering.py \
#    --prompt_type="base" \
#    --model_name="meta-llama/Meta-Llama-3.1-8B" \
#    --cache_dir=$CACHE_DIR \
#    --access_token=$ACCESS_TOKEN \
#    --input_file="data/splits/lextime_512samples.csv" \
#    --output_file="output.csv"  
