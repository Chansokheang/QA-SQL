# Enhanced Retrieval for RAG-to-SQL

This document explains the enhanced retrieval features implemented to improve SQL generation accuracy.

## New Features

1. **Cross-Encoder Reranking**
   - Uses a more powerful cross-encoder model to rerank initial retrieval results
   - Combines vector similarity with semantic relevance for better retrieval

2. **Query Decomposition**
   - Breaks down complex queries into simpler sub-queries
   - Performs retrieval on each sub-query and combines results
   - Improves handling of multi-part or complex questions

3. **Fine-Tuned Embeddings**
   - Customizes embedding models specifically for SQL-related text
   - Better captures the relationship between natural language questions and SQL queries

## How to Use

### Installing Dependencies

First, install the required packages:

```bash
pip install -r requirements.txt
```

### Fine-tuning Embeddings (Optional)

To fine-tune a custom embedding model:

```bash
python -m src.finetune_embeddings --epochs 10 --batch_size 16
```

This will:
1. Load sample SQL questions and queries from your dataset
2. Fine-tune a sentence transformer model on this data
3. Save the model to `fine_tuned_sql_embeddings/` directory

### Running with Enhanced Retrieval

Use the enhanced features with the `run.py` script:

```bash
# Basic enhanced retrieval with cross-encoder reranking
python run.py --retriever enhanced --llm claude

# Adding query decomposition
python run.py --retriever enhanced --decompose --llm claude

# Using fine-tuned embeddings
python run.py --retriever enhanced --finetuned-embeddings --llm claude

# Combining all features
python run.py --retriever enhanced --decompose --finetuned-embeddings --llm claude
```

## Parameter Details

- `--retriever enhanced`: Use the enhanced hybrid retriever with cross-encoder reranking
- `--decompose`: Enable query decomposition for complex questions
- `--finetuned-embeddings`: Use the fine-tuned embedding model (requires running finetune_embeddings.py first)
- `--llm [openai|claude|groq]`: Choose which LLM to use (OpenAI, Claude, or Groq)
- `--hyde` / `--no-hyde`: Enable/disable HyDE (Hypothetical Document Embeddings)

## Implementation Details

### EnhancedHybridRetriever

This retriever combines three techniques:
1. Vector similarity search using Chroma DB
2. Keyword-based search with Jaccard similarity
3. Cross-encoder reranking using a more powerful model

The process:
1. Retrieve more documents than needed (default: k=10)
2. Apply both vector and keyword search (similar to the original HybridRetriever)
3. Rerank results using a cross-encoder model
4. Return the top k results (default: k=5)

### Query Decomposition

When enabled:
1. The complex query is sent to the LLM to be broken down into 2-3 simpler sub-queries
2. Each sub-query is processed by the retriever separately
3. Results are combined, deduplicated, and reranked based on relevance to the original query
4. The top k most relevant results are returned

### Fine-Tuned Embeddings

The fine-tuning process:
1. Uses contrastive learning to train a model to bring questions and their SQL representations closer in the embedding space
2. Creates a custom `FineTunedEmbeddings` class that can be used as a drop-in replacement for `OpenAIEmbeddings`
3. Improves semantic understanding specific to SQL queries and database questions

## Performance Notes

- Cross-encoder reranking significantly improves retrieval relevance but is computationally more expensive
- Query decomposition is especially helpful for complex, multi-part questions
- Fine-tuned embeddings can improve performance on domain-specific databases

To get the best performance:
1. For simple questions: Use basic hybrid retrieval
2. For complex questions: Enable query decomposition
3. For best overall results: Use enhanced retrieval with query decomposition and fine-tuned embeddings