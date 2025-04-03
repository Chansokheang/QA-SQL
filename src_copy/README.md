# SQL Sample Generation Tool

This tool generates SQL samples for question-answering benchmarks using various large language models (LLMs).

## Overview

The `sample_generation.py` script creates SQL sample questions and corresponding queries based on provided database schemas and evidence. It retrieves relevant evidence from a vector database and uses LLMs to generate high-quality, diverse SQL questions.

## Features

- Generate SQL samples with different LLMs (Claude, OpenAI, Groq)
- Customize number of samples per database
- Configure output directory
- Evidence-based SQL generation with vector search

## Prerequisites

- Python 3.8+
- Required Python packages:
  - anthropic
  - openai
  - langchain_community
  - langchain_groq
  - langchain_chroma
  - langchain_openai
  - sqlalchemy
- API keys for the LLMs you want to use

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install anthropic openai langchain-community langchain-groq langchain-chroma langchain-openai sqlalchemy
   ```
3. Set up environment variables for API keys:
   ```
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GROQ_API_KEY="your-groq-key"
   ```

## Usage

### Sample Generation

```bash
python sample_generation.py --llm [claude|openai|groq] --samples [NUM_SAMPLES] --output [OUTPUT_DIR]
```

#### Parameters

- `--llm`: LLM to use for generation (required, choices: claude, openai, groq)
- `--samples`: Number of samples to generate per database (default: 25)
- `--output`: Output directory path (default: ./sample)

#### Examples

Generate 25 samples using Claude:
```bash
python sample_generation.py --llm claude
```

Generate 50 samples using OpenAI with custom output directory:
```bash
python sample_generation.py --llm openai --samples 50 --output ./output/openai_samples
```

Generate samples using Groq:
```bash
python sample_generation.py --llm groq --output ./output/groq_samples
```

### Sample Ingestion

Ingest generated samples into a Chroma vector database for retrieval:

```bash
python sample_ingestion.py --dataset [DATASET_PATH] --persistent_dir [CHROMA_DB_DIR]
```

#### Parameters

- `--dataset`: Path to the dataset JSON file (default: ./sample/Groq_generated_sample.json)
- `--persistent_dir`: Directory to persist Chroma DB (default: ../database/chroma/chroma_db)

#### Examples

Ingest a custom dataset:
```bash
python sample_ingestion.py --dataset ./output/claude_samples.json
```

Specify both dataset and persistent directory:
```bash
python sample_ingestion.py --dataset ./sample/openai_generated_sample.json --persistent_dir ./my_chroma_db
```

### Running RAG-to-SQL

Run the RAG-to-SQL approach on a dataset using your preferred LLM:

```bash
python run.py --persistent_dir [CHROMA_DB_DIR] --output_dir [OUTPUT_DIR] --llm [openai|claude|groq] --input [INPUT_DATASET]
```

#### Parameters

- `--persistent_dir`: Directory where the Chroma DB is stored (default: ./database/chroma/chroma_db)
- `--output_dir`: Directory to save output files (default: ./outputs)
- `--llm`: LLM to use for SQL generation (choices: openai, claude, groq; default: groq)
- `--input`: Path to input dataset JSON file (default: ./database/dev_folder/dev.json)

#### Examples

Run with default settings (Groq LLM):
```bash
python run.py
```

Use Claude with a custom Chroma DB:
```bash
python run.py --llm claude --persistent_dir ./my_chroma_db
```

Generate SQL with OpenAI and save to a custom output directory:
```bash
python run.py --llm openai --output_dir ./results/openai_results
```

Process a custom dataset:
```bash
python run.py --input ./my_dataset.json --llm groq
```

## Output Format

The script generates a JSON file with the following structure:

```json
[
  {
    "question_id": 0,
    "db_id": "database_name",
    "question": "What is the total amount...",
    "evidence": "Evidence that justifies the SQL query",
    "SQL": "SELECT SUM(amount) FROM transactions WHERE account_id = 1;",
    "difficulty": "simple"
  },
  ...
]
```

## File Organization

- The output file is saved in the specified output directory with the naming pattern: `{llm_choice}_generated_sample.json`
- Each question is assigned a unique `question_id`