import argparse
import json
import os
import traceback
import importlib
from src import utils

def setup_rag_chain(persistent_dir, llm_choice, retriever_type="hybrid", use_hyde=True, 
                    use_enhanced_retrieval=False, use_query_decomposition=False,
                    use_finetuned_embeddings=False):
    """
    Dynamically configure and import the RAG chain with custom settings
    
    Args:
        persistent_dir: Directory where the Chroma DB is stored
        llm_choice: LLM to use (openai, claude, or groq)
        retriever_type: Type of retriever to use (hybrid or vector)
        use_hyde: Whether to use HyDE for retrieval
        use_enhanced_retrieval: Whether to use enhanced retrieval with reranking
        use_query_decomposition: Whether to use query decomposition
        use_finetuned_embeddings: Whether to use fine-tuned embeddings
        
    Returns:
        The configured RAG_to_SQL chain
    """
    # Set environment variables for the RAG_to_SQL module
    os.environ["RAG2SQL_PERSISTENT_DIR"] = persistent_dir
    os.environ["RAG2SQL_LLM_CHOICE"] = llm_choice
    os.environ["RAG2SQL_RETRIEVER_TYPE"] = retriever_type
    os.environ["RAG2SQL_USE_HYDE"] = str(use_hyde).lower()
    os.environ["RAG2SQL_USE_ENHANCED_RETRIEVAL"] = str(use_enhanced_retrieval).lower()
    os.environ["RAG2SQL_USE_QUERY_DECOMPOSITION"] = str(use_query_decomposition).lower()
    os.environ["RAG2SQL_USE_FINETUNED_EMBEDDINGS"] = str(use_finetuned_embeddings).lower()
    
    # Re-import module to pick up environment variables
    RAG_to_SQL = importlib.import_module("src.RAG_to_SQL")
    
    # Reload to ensure we get fresh module with new settings
    importlib.reload(RAG_to_SQL)
    
    return RAG_to_SQL.chain

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RAG to SQL conversion on a dataset")
    parser.add_argument("--persistent_dir", type=str, 
                        help="Directory where the Chroma DB is stored",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           "database", "chroma", "chroma_db"))
    parser.add_argument("--output_dir", type=str, 
                        help="Directory to save output files",
                        default="./outputs")
    parser.add_argument("--llm", type=str, choices=["openai", "claude", "groq"], 
                        help="LLM to use for SQL generation", default="groq")
    parser.add_argument("--input", type=str, 
                        help="Path to input dataset JSON file",
                        default="./database/dev_folder/dev.json")
    parser.add_argument("--retriever", type=str, choices=["vector", "hybrid", "enhanced"],
                        help="Retriever type to use", default="hybrid")
    parser.add_argument("--hyde", action="store_true", default=True,
                        help="Use HyDE for retrieval")
    parser.add_argument("--no-hyde", dest="hyde", action="store_false",
                        help="Disable HyDE for retrieval")
    parser.add_argument("--decompose", action="store_true", default=False,
                        help="Use query decomposition for complex queries")
    parser.add_argument("--finetuned-embeddings", action="store_true", default=False,
                        help="Use fine-tuned embeddings (requires running finetune_embeddings.py first)")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if using enhanced retrieval
    use_enhanced_retrieval = args.retriever == "enhanced"
    
    # For backward compatibility, set retriever_type to hybrid if enhanced was selected
    retriever_type = "hybrid" if use_enhanced_retrieval else args.retriever
    
    # Define output filename based on configuration
    retriever_suffix = args.retriever
    hyde_suffix = "hyde" if args.hyde else "nohydе"
    decompose_suffix = "decomp" if args.decompose else "nodecomp"
    embeddings_suffix = "finetuned" if args.finetuned_embeddings else "default"
    
    output_file = os.path.join(args.output_dir, 
                               f"{args.llm}_{retriever_suffix}_{hyde_suffix}_{decompose_suffix}_{embeddings_suffix}.json")
    
    # Load dataset
    print(f"Loading dataset from {args.input}")
    with open(args.input, 'r') as f:
        datasets = json.load(f)
    
    # Setup RAG chain with specified settings
    chain = setup_rag_chain(
        persistent_dir=args.persistent_dir, 
        llm_choice=args.llm,
        retriever_type=retriever_type,
        use_hyde=args.hyde,
        use_enhanced_retrieval=use_enhanced_retrieval,
        use_query_decomposition=args.decompose,
        use_finetuned_embeddings=args.finetuned_embeddings
    )
    
    # Process questions
    question_list, db_name_list = utils.decouple_question(datasets)
    responses = []
    
    results = {}
    print(f"Starting processing of {len(question_list)} questions with configuration:")
    print(f"  - LLM: {args.llm}")
    print(f"  - Retriever: {args.retriever}")
    print(f"  - Using HyDE: {args.hyde}")
    print(f"  - Using query decomposition: {args.decompose}")
    print(f"  - Using fine-tuned embeddings: {args.finetuned_embeddings}")
    print(f"  - Using Chroma DB from: {args.persistent_dir}")
    print(f"  - Output will be saved to: {output_file}")
    
    for i, (question, db_name) in enumerate(zip(question_list, db_name_list)):
        try:
            print(f"Processing question {i}: {question}")
            sql = chain.invoke({
                "question": question,
                "db_name": db_name
            })
            results[i] = sql  # Store successful result
            print(f"✅ Generated SQL: {sql[:80]}..." if len(sql) > 80 else f"✅ Generated SQL: {sql}")

        except Exception as e:
            print(f"⚠️ Error processing question {i}: {question}")
            print(traceback.format_exc())  # Print detailed error traceback
            results[i] = f"Error: {str(e)}"  # Save error message instead of SQL

        # Save partial results immediately to prevent loss if interrupted
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        # Print progress
        progress = (i + 1) / len(question_list) * 100
        print(f"Progress: {progress:.1f}% ({i+1}/{len(question_list)})")

    ordered_results = [results[i] for i in sorted(results.keys())]
    for i, result in enumerate(ordered_results):
        print(f"Question {i} SQL Response: {result[:100]}..." if len(result) > 100 else f"Question {i} SQL Response: {result}")   
        responses.append(result+'\t----- bird -----\t'+db_name_list[i])
    
    utils.generate_sql_file(responses, output_file)
    print(f"✅ Processing complete! Results saved to {output_file}")
    