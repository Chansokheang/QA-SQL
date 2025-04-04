import argparse
import json
import os
import traceback
import importlib
from src import utils

def setup_rag_chain(persistent_dir, llm_choice, use_hyde=True):
    """
    Dynamically configure and import the RAG chain with custom settings
    """
    # Set environment variables for the RAG_to_SQL module
    os.environ["RAG2SQL_PERSISTENT_DIR"] = persistent_dir
    os.environ["RAG2SQL_LLM_CHOICE"] = llm_choice
    os.environ["RAG2SQL_USE_HYDE"] = "true" if use_hyde else "false"
    
    # Re-import module to pick up environment variables
    RAG_to_SQL = importlib.import_module("src.RAG_to_SQL")
    
    # Reload to ensure we get fresh module with new settings
    importlib.reload(RAG_to_SQL)
    
    return RAG_to_SQL.chain

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RAG to SQL conversion on a dataset in batches")
    parser.add_argument("--persistent_dir", type=str, 
                        help="Directory where the Chroma DB is stored",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           "database", "chroma", "chroma_db"))
    parser.add_argument("--output_dir", type=str, 
                        help="Directory to save output files",
                        default="./outputs")
    parser.add_argument("--llm", type=str, choices=["openai", "claude", "groq"], 
                        help="LLM to use for SQL generation", default="openai")
    parser.add_argument("--input", type=str, 
                        help="Path to input dataset JSON file",
                        default="./database/dev_folder/dev.json")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting index for batch processing (default: 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="Ending index for batch processing (exclusive, default: None)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size if end index is not specified (default: 10)")
    parser.add_argument("--no-hyde", action="store_true",
                        help="Disable HyDE for context retrieval")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file instead of creating a new one")
    args = parser.parse_args()
     
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output filename based on LLM choice and hyde usage
    hyde_suffix = "" if not args.no_hyde else "_no_hyde"
    output_file = os.path.join(args.output_dir, f"{args.llm}{hyde_suffix}_0404_no_hyde_chroma.json")
    
    # Load dataset
    print(f"Loading dataset from {args.input}")
    with open(args.input, 'r') as f:
        datasets = json.load(f)
    
    # Process questions
    question_list, db_name_list = utils.decouple_question(datasets)
    
    # Setup end index if not specified
    if args.end is None:
        args.end = min(args.start + args.batch_size, len(question_list))
    else:
        args.end = min(args.end, len(question_list))
    
    # Validate indices
    if args.start < 0 or args.start >= len(question_list):
        print(f"Error: Start index {args.start} is out of range. Dataset has {len(question_list)} questions.")
        exit(1)
    
    if args.end <= args.start:
        print(f"Error: End index {args.end} must be greater than start index {args.start}.")
        exit(1)
    
    # Check if output file exists and we're not appending
    if os.path.exists(output_file) and not args.append:
        print(f"Warning: Output file {output_file} already exists and --append is not set.")
        confirmation = input("Do you want to overwrite the file? (y/n): ")
        if confirmation.lower() != 'y':
            print("Exiting without processing.")
            exit(0)
    
    # Setup RAG chain with specified settings
    chain = setup_rag_chain(args.persistent_dir, args.llm, not args.no_hyde)
    
    # Load existing results if appending
    results = {}
    if args.append and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {output_file}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            print("Starting with empty results.")
    
    print(f"Processing questions {args.start} to {args.end-1} out of {len(question_list)} with {args.llm} LLM")
    print(f"HyDE for context retrieval: {'Disabled' if args.no_hyde else 'Enabled'}")
    print(f"Using Chroma DB from {args.persistent_dir}")
    print(f"Output will be saved to {output_file}")
    
    for i in range(args.start, args.end):
        question = question_list[i]
        db_name = db_name_list[i]
        
        try:
            print(f"Processing question {i}: {question}")
            sql = chain.invoke({
                "question": question,
                "db_name": db_name
            })
            results[str(i)] = sql  # Store successful result
            print(f"✅ Generated SQL: {sql[:80]}..." if len(sql) > 80 else f"✅ Generated SQL: {sql}")

        except Exception as e:
            print(f"⚠️ Error processing question {i}: {question}")
            print(traceback.format_exc())  # Print detailed error traceback
            results[str(i)] = f"Error: {str(e)}"  # Save error message instead of SQL

        # Save partial results immediately to prevent loss if interrupted
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        # Print progress
        progress = (i - args.start + 1) / (args.end - args.start) * 100
        print(f"Batch progress: {progress:.1f}% ({i-args.start+1}/{args.end-args.start})")
        print(f"Overall progress: {(i + 1) / len(question_list) * 100:.1f}% ({i+1}/{len(question_list)})")
    
    # Always generate BIRD format for all processed entries so far
    try:
        # Convert ALL processed results to BIRD format
        bird_file = os.path.join(args.output_dir, f"{args.llm}{hyde_suffix}_bird_0404_no_hyde_chroma.json")
        
        # First, load existing BIRD format results if appending
        existing_bird_results = {}
        if args.append and os.path.exists(bird_file):
            try:
                with open(bird_file, 'r') as f:
                    existing_bird_results = json.load(f)
                print(f"Loaded {len(existing_bird_results)} existing BIRD format results")
            except Exception as e:
                print(f"Error loading existing BIRD results: {e}")
        
        # Convert ALL processed results to ordered list with BIRD format
        responses = []
        # Process all entries that have been processed so far
        for i in range(len(question_list)):
            if str(i) in results:
                responses.append((i, results[str(i)] + '\t----- bird -----\t' + db_name_list[i]))
        
        # Generate final output file
        utils.generate_sql_file(responses, bird_file, append=False)  # Always overwrite with complete results
        print(f"✅ BIRD format results saved to {bird_file}")
    except Exception as e:
        print(f"Error creating BIRD format output: {e}")
        print(traceback.format_exc())
    
    print(f"✅ Batch processing complete! Raw results saved to {output_file}")
    print("\nTo continue processing the next batch, run:")
    next_start = args.end
    next_end = min(next_start + args.batch_size, len(question_list))
    command = f"python run2.py --llm {args.llm} --start {next_start} --end {next_end}"
    
    if args.no_hyde:
        command += " --no-hyde"
    
    command += " --append"
    
    print(f"\n{command}")