#!/bin/bash

# Define an array of scripts to run
scripts=("prepare_data.py" "fine_tune.py" "evaluate.py" "extract_embs_and_plot.py")
#scripts=("evaluate.py" "extract_embs_and_plot.py")

# Loop through each script and execute
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python3 "$script"
done
