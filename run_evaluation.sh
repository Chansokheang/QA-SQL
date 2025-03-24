#!/bin/bash

# Configuration
db_root_path='/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/database/dev_folder/dev_databases/'
data_mode='dataset'
diff_json_path='/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/outputs/dev.json'
predicted_sql_path_kg='/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/outputs/'
ground_truth_path='./outputs/'
num_cpus=16
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'
iterate_num=100  # For VES evaluation

# Get file_name from command line argument
file_name="$1"
eval_type="$2"  # Optional: can be "acc" (default), "ves", or "both"

# Set default evaluation type if not specified
if [ -z "$eval_type" ]; then
    eval_type="both"
fi

# Validate input
if [ -z "$file_name" ]; then
    echo "Usage: ./run_evaluation.sh <file_name> [eval_type]"
    echo "Example: ./run_evaluation.sh groq.json both"
    echo "eval_type options: acc, ves, both (default)"
    echo ""
    echo "No file name provided, will use default (claude.json)"
    acc_cmd="python3 -u ./evaluation/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
    --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
    --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}"
    
    ves_cmd="python3 -u ./evaluation/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
    --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
    --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} --iterate_num ${iterate_num}"
else
    # Check if the file exists
    if [ ! -f "${predicted_sql_path_kg}${file_name}" ]; then
        echo "Error: File ${predicted_sql_path_kg}${file_name} not found!"
        exit 1
    fi
    
    echo "Evaluating using file: ${file_name}"
    acc_cmd="python3 -u ./evaluation/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
    --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
    --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} --file_name ${file_name}"
    
    ves_cmd="python3 -u ./evaluation/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
    --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
    --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} --iterate_num ${iterate_num} --file_name ${file_name}"
fi

# Run evaluations based on type
if [ "$eval_type" = "acc" ] || [ "$eval_type" = "both" ]; then
    echo "Starting accuracy evaluation..."
    $acc_cmd
fi

if [ "$eval_type" = "ves" ] || [ "$eval_type" = "both" ]; then
    echo "Starting VES evaluation..."
    $ves_cmd
fi

echo "Evaluation completed!"