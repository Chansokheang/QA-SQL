# RAG-to-SQL Benchmark

This is a comprehensive benchmark for evaluating RAG-to-SQL approaches, which use retrieval-augmented generation to produce SQL queries from natural language questions.

## Installation

### Prerequisites

- Python 3.8+
- API keys for LLMs (OpenAI, Anthropic, Groq)

### Setup Steps

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Set up database files
   ```bash
   # Make the database setup script executable
   chmod +x database/setup/get_dbs.sh
   
   # Run the database setup script
   ./database/setup/get_dbs.sh
   ```

3. Set up environment variables for API keys
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GROQ_API_KEY="your-groq-key"
   ```

4. Make the evaluation script executable
   ```bash
   chmod +x run_evaluation.sh
   ```

   Run the evaluations using the provided script:

   ```bash
   # Run both accuracy and VES evaluations (recommended order)
   ./run_evaluation.sh [file_name] both

   # Run only accuracy evaluation
   ./run_evaluation.sh [file_name] acc

   # Run only VES evaluation (should be run after accuracy evaluation)
   ./run_evaluation.sh [file_name] ves
   ```

   Where:
   - `file_name`: Name of the prediction file to evaluate (e.g., 'claude.json', 'gpt4o.json')

   Examples:

   ```bash
   # Run both accuracy and VES evaluation on claude.json
   ./run_evaluation.sh claude.json

   # Run only accuracy evaluation on gpt4o.json
   ./run_evaluation.sh gpt4o.json acc

   # Run only VES evaluation on groq.json
   ./run_evaluation.sh groq.json ves
   ```

   **Note**: It's recommended to run accuracy evaluation first, and then VES evaluation, as VES only measures performance for correct queries.

## Project Overview

RAG-to-SQL combines retrieval augmented generation (RAG) with SQL query generation. The system:
1. Retrieves relevant examples from a vector database containing SQL samples
2. Uses these examples to guide large language models in generating accurate SQL queries for new natural language questions

## Repository Structure

- `src/`: Source code for sample generation, ingestion, and RAG-to-SQL implementation
- `evaluation/`: Evaluation scripts for accuracy and execution performance
- `database/`: Database files and schemas
- `outputs/`: Generated query outputs and evaluation results

## Features

- Generate SQL samples with different LLMs (Claude, OpenAI, GPT-4o, Groq)
- Evaluate SQL queries for correctness and execution efficiency
- Customize evaluation parameters and LLM selection
- Retrieve contextually relevant examples from a vector store

## Usage

### Extracting Evidence and Creating Evidence Vector Database

The benchmark includes a tool to extract evidence from the development dataset and create a persistent Chroma vector database in the evidence directory:

```bash
python src/evidence_extractor.py --dev_file [DEV_FILE_PATH] --output_dir [EVIDENCE_DIR] --chroma_dir [CHROMA_DB_DIR]
```

Parameters:
- `dev_file`: Path to the dev.json file (default: ../database/dev_folder/dev.json)
- `output_dir`: Directory to store evidence files (default: ./evidence)
- `chroma_dir`: Directory to store the Chroma database (default: ../database/chroma/evidence_chroma_db)

Example:
```bash
# Use default paths
python src/evidence_extractor.py

# Specify custom paths
python src/evidence_extractor.py --dev_file ./custom/dev.json --output_dir ./custom_evidence --chroma_dir ./custom_db
```

### Generating Samples

```bash
python src/sample_generation.py --llm [claude|openai|groq] --samples [NUM_SAMPLES] --output [OUTPUT_DIR]
```

### Ingesting Samples into Vector Database

```bash
python src/sample_ingestion.py --dataset [DATASET_PATH] --persistent_dir [CHROMA_DB_DIR]
```

### Running RAG-to-SQL

```bash
python run.py --persistent_dir [CHROMA_DB_DIR] --output_dir [OUTPUT_DIR] --llm [openai|claude|groq] --input [INPUT_DATASET]
```

### Evaluation

The benchmark includes two types of evaluation that should be run in sequence:

1. **Accuracy Evaluation**: Measures how many generated SQL queries produce the correct results when executed
2. **VES (Velocity Efficiency Score) Evaluation**: Measures the execution efficiency of the generated queries compared to the ground truth queries

For details on running evaluations, see step 5 in the Installation section above.

## Evaluation Metrics

### Accuracy

The accuracy metric measures whether the generated SQL query produces the same results as the ground truth query. Results are broken down by difficulty level (simple, moderate, challenging).

### VES (Velocity Efficiency Score)

VES measures the execution efficiency of generated SQL queries compared to ground truth. The score is calculated as:

```
VES = sqrt(t_gt / t_pred) * 100
```

Where:
- `t_gt` is the execution time of the ground truth query
- `t_pred` is the execution time of the predicted query

A higher VES indicates better performance, with 100 being equal to the ground truth.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.