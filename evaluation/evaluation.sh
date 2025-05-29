#!/bin/bash

# Exit on error
set -e

# Define base folder and models
BASE_FOLDER="results"
MODELS="gemini-2.5-pro-preview-05-06"  # Add more model names to this list if needed

# Activate virtual environment if needed
# source venv/bin/activate

# Run the evaluation script
python3 evaluation/evaluation.py --base_folder "$BASE_FOLDER" --models "${MODELS}"

# chmod +x evaluation/evaluation.sh