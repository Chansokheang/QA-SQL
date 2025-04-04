#!/bin/bash

# Default starting index is 0 if not provided
START_IDX=${1:-0}
BATCH_SIZE=${2:-10}

# Static parameters - configured for your specific setup

# PREDICTED_SQL_PATH="/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/outputs/test_bird/openai_no_hyde_bird_0404_no_hyde.json"
PREDICTED_SQL_PATH="/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/outputs/test_bird/openai_bird_0304.json"
# PREDICTED_SQL_PATH="/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/outputs/groq.json"
GROUND_TRUTH_PATH="/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/database/dev_folder/"
DATA_MODE="dev"
DB_ROOT_PATH="/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/database/dev_folder/dev_databases/"
DIFF_JSON_PATH="/home/sokheang/Research/RAG2SQL/RAG_2_SQL_Benchmark/mini_dev/database/dev_folder/dev.json"

# Run evaluation batch
python evaluation/evaluation_batch.py \
  --predicted_sql_path="$PREDICTED_SQL_PATH" \
  --ground_truth_path="$GROUND_TRUTH_PATH" \
  --data_mode="$DATA_MODE" \
  --db_root_path="$DB_ROOT_PATH" \
  --diff_json_path="$DIFF_JSON_PATH" \
  --start_idx="$START_IDX" \
  --batch_size="$BATCH_SIZE"