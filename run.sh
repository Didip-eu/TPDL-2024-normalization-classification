#!/bin/bash

classifiers=("lr" "nb" "svm" "svm+" "xgb" "deberta" "roberta")
embed_columns=("text" "text_normalized")
tasks=("dating" "locating")

base_cmd="python ./src/run_refactored.py"

for task in "${tasks[@]}"; do
    for embed_column in "${embed_columns[@]}"; do
        for classifier in "${classifiers[@]}"; do
            
            cmd="$base_cmd --classifier $classifier --embed_column $embed_column --task $task"
            
            if [[ "$classifier" == "deberta" || "$classifier" == "roberta" ]]; then
                cmd="$cmd --batch_size 12 --max_length 512 --accum_steps 1"
            fi
            
            echo "Running: $cmd"
            eval $cmd
            
            sleep 1
        done
    done
done

echo "All experiments completed."