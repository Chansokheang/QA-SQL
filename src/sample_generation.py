import os
import json
import argparse
import sqlite3
import anthropic
from openai import OpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from utils import groq_extract_content, claude_extract_content, ollama_extract_content, openai_extract_content, create_persistent_db, extract_evidence
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# def is_sql_correct(schema: str, question: str, sql: str, llm_choice="openai") -> bool:
#     """
#     Check if the generated SQL query accurately answers the question based on the schema.
#
#     Args:
#         schema: Database schema information
#         question: User's question
#         sql: Generated SQL query
#         llm_choice: The LLM to use for validation
#
#     Returns:
#         bool: True if the SQL is correct, False otherwise
#     """
#     validation_prompt = f"""
#     Database Schema:
#     {schema}
#
#     Question: {question}
#     Generated SQL: {sql}
#
#     Does the Generated SQL query accurately and completely answer the Question based *only* on the provided Database Schema? Answer strictly with YES or NO.
#     """
#
#     messages = [
#         {
#             "role": "user",
#             "content": validation_prompt
#         }
#     ]
#
#     try:
#         client = OpenAI()
#         if llm_choice == "openai":
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 max_tokens=1024,
#                 temperature=0
#             )
#             answer = response.choices[0].message.content.strip().upper()
#         elif llm_choice == "claude":
#             client = anthropic.Anthropic()
#             response = client.messages.create(
#                 model="claude-3-7-sonnet-20250219",
#                 temperature=0,
#                 max_tokens=1000,
#                 messages=messages
#             )
#             answer = response.content[0].text.strip().upper()
#         elif llm_choice == "groq":
#             llm = ChatGroq(
#                 model="Llama-3.3-70b-Versatile",
#                 temperature=0.0,
#                 groq_api_key=os.environ.get("GROQ_API_KEY")
#             )
#             response = llm.invoke(messages)
#             if hasattr(response, "content"):
#                 answer = response.content.strip().upper()
#             else:
#                 logging.error("LLM response does not contain a valid 'content' attribute.")
#                 return False  # Default to invalid if validation fails
#         else:
#             logging.error(f"Unsupported LLM choice: {llm_choice}")
#             return False  # Default to invalid if LLM choice is invalid
#
#         return "YES" in answer
#
#     except Exception as e:
#         logging.error(f"Error during SQL validation: {e}")
#         return False  # Default to invalid if any error occurs

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Constants
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database/dev_folder/dev_databases"))
DEV_FOLDER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database/dev_folder"))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(CURRENT_DIR, "evidence")
DATABASE_DIR = os.path.join(CURRENT_DIR, "..", "database")
CHROMA_DIR = os.path.join(DATABASE_DIR, "chroma")
PERSISTENT_DIR = os.path.join(CHROMA_DIR, "chroma_db")
EVIDENCE_DIR = os.path.join(CHROMA_DIR, "evidence_chroma_db")
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

# Client initialization happens on-demand in each LLM function

def invoke_claude(messages):
    """
    Calls Claude to generate or refine responses.
    """
    client = anthropic.Anthropic()
    try:
        system_message = "You are an SQL generation assistant. Your task is to generate a JSON structure containing questions and SQL queries based on the provided database schema and relevant evidence."
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            temperature=0,
            max_tokens=4000,
            messages=messages,
            system=system_message,
        )
        
        result = response.content[0].text.strip() if response.content else ""
        print(f"Claude response received, length: {len(result)}")
        return result
    except Exception as e:
        print(f"Error invoking Claude: {e}")
        return ""

def invoke_openai(messages):
    """
    Calls OpenAI to generate responses.
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip() if response.choices else ""

def invoke_groq(messages):
    """
    Calls Groq to generate responses.
    """
    llm = ChatGroq(
        model="Llama-3.3-70b-Versatile",
        temperature=0.7,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    response = llm.invoke(messages)
    return response

def generate_sample(llm_choice, num_sample, output_dir):
    try:
        vector_store = Chroma(
            collection_name="evidence_collection",
            persist_directory=EVIDENCE_DIR,
            embedding_function=embedding_function
        )

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5},
        )
        
        # Define the distribution of question types and difficulty levels
        question_types = [
            "exact", "aggregation", "comparison", 
            "ranking", "reasoning", "multi-table", "nested"
        ]
        difficulty_levels = ["simple", "moderate", "challenge"]
        
        # Ensure DB_DIR is a directory
        if not os.path.isdir(DB_DIR):
            raise NotADirectoryError(f"DB_DIR is not a directory: {DEV_FOLDER_DIR}")

        # List the contents if the directory exists
        db_list = os.listdir(DB_DIR)
        db_list = db_list[:-1]
        question_id_counter = 0  
        all_questions = []
        
        for db_name in db_list:
            print(f"Processing database: {db_name}")
            db_path = f'{DB_DIR}/{db_name}/{db_name}.sqlite'
            db_path = os.path.abspath(db_path)
            if not os.path.exists(db_path):
                print(f"Error: Database file does not exist at path {db_path}")
                continue  # Skip to the next database
                
            # Load Database Schema
            engine = create_engine(f"sqlite:///{db_path}")
            db = SQLDatabase(engine)
            schemas = db.get_table_info()

            # Retrieve Evidence from Vector Database
            docs = retriever.invoke(f"Retrieve evidence related to SQL queries for the {db_name} database.")

            if not docs:
                print(f"⚠️ No relevant evidence found for {db_name}!")
            else:
                print(f"✅ Retrieved {len(docs)} documents for {db_name}:")
                for i, doc in enumerate(docs):
                    print(f"\n📄 Document {i+1}: {doc.metadata}")
                    print(f"📝 Content: {doc.page_content[:200]}...")

            evidence_list = [doc.page_content for doc in docs]
            combined_evidence = "\n".join(evidence_list)
            
            # Common prompt content for all LLMs
            prompt_content = f"""
                **Database Schema:**
                {schemas}

                **Relevant Evidence:**
                {combined_evidence}

                **Instructions:**
                - Generate **unique questions** in natural language using the schema and evidence.
                - For each question, generate a corresponding **valid SQL query** based on the schema and evidence. Do not include any markdown or code block markers like ```json..
                - Create a variety of question types:
                    - **Exact**: Direct retrieval of specific values
                    - **Aggregation**: Using functions like COUNT, AVG, SUM, MIN, MAX
                    - **Comparison**: Using values under conditions
                    - **Ranking**: Using ORDER BY with LIMIT to get top/bottom results
                    - **Reasoning**: Questions requiring logical reasoning with joins or complex conditions
                    - **Multi-table**: Questions requiring joins across multiple tables
                    - **Nested**: Questions requiring subqueries or nested operations
                - For each type, create questions with varying difficulties from simple to challenging
                - Ensure the output is in JSON format with the following fields:
                    - `question_id`: A unique numeric ID for each question.
                    - `db_id`: The name of the database (e.g., "{{db_name}}").
                    - `question`: The question in natural language.
                    - `evidence`: Use the provided evidence to justify the SQL query.
                    - `SQL`: A valid SQL query based on the schema and evidence and wrap table name like `order`, `bond`,... .
                    - `difficulty`: A difficulty level for the question ("simple", "moderate", "challenge").
                    - `question_type`: The type of question (exact, aggregation, comparison, ranking, reasoning, multi-table, nested).

                **Example Output:**
                [
                    {{
                        "question_id": 0,
                        "db_id": "{{db_name}}",
                        "question": "What is the total amount of money transferred from account 1 to account 2?",
                        "evidence": "Transferred amount = SUM(`amount`) WHERE `account_id` = 1 AND `account_to` = 2",
                        "SQL": "SELECT SUM(amount) FROM transactions WHERE account_id = 1 AND account_to = 2;",
                        "difficulty": "simple",
                        "question_type": "aggregation"
                    }},
                    {{
                        "question_id": 1,
                        "db_id": "{{db_name}}",
                        "question": "Which customer has made the most transactions, and how many did they make?",
                        "evidence": "Customer transactions can be counted using COUNT(*) GROUP BY customer_id",
                        "SQL": "SELECT c.name, COUNT(*) as transaction_count FROM transactions t JOIN customers c ON t.customer_id = c.id GROUP BY t.customer_id ORDER BY transaction_count DESC LIMIT 1;",
                        "difficulty": "challenge",
                        "question_type": "ranking"
                    }}
                ]

                Generate {num_sample} questions and their corresponding SQL queries without any explanation.
            """
            
            # Process based on LLM choice
            if llm_choice == "claude":
                messages = [
                    {
                        "role": "user", 
                        "content": prompt_content.strip()
                    }
                ]
                response = invoke_claude(messages)
                print(f"Calling claude_extract_content with response length: {len(response)}")
                json_data = claude_extract_content(response)
                
            elif llm_choice == "openai":
                messages = [
                    {
                        "role": "system", 
                        "content": "You are an SQL generation assistant. Your task is to generate a JSON structure containing questions and SQL queries based on the provided database schema and relevant evidence."
                    },
                    {
                        "role": "user", 
                        "content": prompt_content.strip()
                    }
                ]
                response = invoke_openai(messages)
                json_data = openai_extract_content(response)
                
            elif llm_choice == "groq":
                messages = [
                    {
                        "role": "system", 
                        "content": "You are an SQL generation assistant. Your task is to generate a JSON structure containing questions and SQL queries based on the provided database schema and relevant evidence."
                    },
                    {
                        "role": "user", 
                        "content": prompt_content.strip()
                    }
                ]
                response = invoke_groq(messages)
                json_data = groq_extract_content(response)
                
            else:
                raise ValueError(f"Unsupported LLM choice: {llm_choice}")
        
            # Ensure response is a valid list
            if isinstance(json_data, list) and json_data:
                print(f"Processing {len(json_data)} questions from {db_name}")
                
                # Check and fix question types and difficulty levels if needed
                for question in json_data:
                    # Set default question_type if missing or invalid
                    if not question.get("question_type") or question.get("question_type") not in question_types:
                        # Guess question type based on SQL content
                        sql = question.get("SQL", "").lower()
                        if "sum(" in sql or "count(" in sql or "avg(" in sql or "min(" in sql or "max(" in sql:
                            question["question_type"] = "aggregation"
                        elif " join " in sql and "where" in sql:
                            question["question_type"] = "multi-table"
                        elif "order by" in sql and "limit" in sql:
                            question["question_type"] = "ranking"
                        elif "select" in sql and "from" in sql and "where" in sql and "select" in sql.split("where")[1]:
                            question["question_type"] = "nested"
                        elif ">" in sql or "<" in sql or "=" in sql:
                            question["question_type"] = "comparison"
                        elif " join " in sql:
                            question["question_type"] = "reasoning"
                        else:
                            question["question_type"] = "exact"
                    
                    # Set default difficulty if missing or invalid
                    if not question.get("difficulty") or question.get("difficulty") not in difficulty_levels:
                        # Guess difficulty based on SQL complexity
                        sql = question.get("SQL", "").lower()
                        if "select" in sql and "from" in sql and "where" not in sql and "join" not in sql:
                            question["difficulty"] = "simple"
                        elif "join" in sql or "group by" in sql:
                            question["difficulty"] = "moderate"
                        elif "having" in sql or ("select" in sql and "from" in sql and "where" in sql and "select" in sql.split("where")[1]):
                            question["difficulty"] = "challenge"
                        elif sql.count("select") > 2 or sql.count("join") > 2:
                            question["difficulty"] = "challenge"
                        else:
                            question["difficulty"] = "moderate"
                    
                    # Validate SQL query by executing it against the database
                    try:
                        print(f"Validating SQL query for question: {question.get('question_id', 'new')} - {question['question'][:50]}...")
                        sql_query = question.get("SQL", "")
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute(sql_query)
                        # Just fetch to test execution, no need to store result
                        cursor.fetchall()
                        conn.close()
                        
                        # Mark query as validated
                        question["is_valid"] = True
                        print(f"✅ SQL query validated successfully")
                        
                        # Add to our question collection
                        question["question_id"] = question_id_counter
                        question_id_counter += 1
                        all_questions.append(question)

                        # # Semantic Validation
                        # if is_sql_correct(schemas, question["question"], sql_query, llm_choice):
                        #     print("✅ Semantic validation passed.")
                        #     question["semantic_valid"] = True
                        # else:
                        #     print("❌ Semantic validation failed.")
                        #     question["semantic_valid"] = False
                        #     # Skip adding invalid queries to all_questions
                            
                    except sqlite3.Error as e:
                        print(f"❌ SQL validation error for question {question.get('question', '')[:50]}: {e}")
                        question["is_valid"] = False
                        question["validation_error"] = str(e)
                        # Skip adding invalid queries to all_questions
                
                # Print distribution of question types, difficulty levels and validation status for this database
                type_counts = {}
                difficulty_counts = {}
                validation_counts = {"valid": 0, "invalid": 0}
                
                # Count only questions that were processed through the validation phase
                for q in all_questions:
                    if q.get("db_id") == db_name:  # Only count questions from current db
                        q_type = q.get("question_type", "unknown")
                        difficulty = q.get("difficulty", "unknown")
                        validation = q.get("is_valid", False)
                        
                        type_counts[q_type] = type_counts.get(q_type, 0) + 1
                        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
                        if validation:
                            validation_counts["valid"] += 1
                        else:
                            validation_counts["invalid"] += 1
                
                print(f"Question type distribution for {db_name}:")
                for q_type, count in type_counts.items():
                    print(f"  - {q_type}: {count} questions")
                
                print(f"Difficulty distribution for {db_name}:")
                for difficulty, count in difficulty_counts.items():
                    print(f"  - {difficulty}: {count} questions")
                    
                print(f"Validation results for {db_name}:")
                valid_percent = (validation_counts["valid"] / (validation_counts["valid"] + validation_counts["invalid"])) * 100 if (validation_counts["valid"] + validation_counts["invalid"]) > 0 else 0
                print(f"  - Valid queries: {validation_counts['valid']} ({valid_percent:.1f}%)")
                print(f"  - Invalid queries: {validation_counts['invalid']}")
            else:
                print(f"Warning: No valid data returned for database {db_name}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Check if we have any data before writing the file
        if all_questions:
            # Generate statistics for the full dataset
            overall_type_counts = {}
            overall_difficulty_counts = {}
            
            for q in all_questions:
                q_type = q.get("question_type", "unknown")
                difficulty = q.get("difficulty", "unknown")
                overall_type_counts[q_type] = overall_type_counts.get(q_type, 0) + 1
                overall_difficulty_counts[difficulty] = overall_difficulty_counts.get(difficulty, 0) + 1
            
            # Count valid and invalid queries
            valid_queries = sum(1 for q in all_questions if q.get("is_valid", False))
            invalid_queries = sum(1 for q in all_questions if not q.get("is_valid", True))
            
            # Create dataset summary
            summary = {
                "total_questions": len(all_questions),
                "databases_covered": len(set(q["db_id"] for q in all_questions)),
                "question_types": overall_type_counts,
                "difficulty_levels": overall_difficulty_counts,
                "validation_results": {
                    "valid_queries": valid_queries,
                    "invalid_queries": invalid_queries,
                    "validation_rate": (valid_queries / len(all_questions)) * 100 if all_questions else 0
                }
            }
            
            # Save questions to JSON file
            output_file = os.path.join(output_dir, f"{llm_choice}_generated_sample3.json")
            with open(output_file, "w") as f:
                json.dump(all_questions, f, indent=4)
            
            # Save summary to separate JSON file
            summary_file = os.path.join(output_dir, f"{llm_choice}_dataset_summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=4)
                
            # Print summary statistics
            print("\n=== DATASET SUMMARY ===")
            print(f"Total questions: {len(all_questions)}")
            print(f"Databases covered: {len(set(q['db_id'] for q in all_questions))}")
            
            print("\nQuestion type distribution:")
            for q_type, count in overall_type_counts.items():
                percentage = (count / len(all_questions)) * 100
                print(f"  - {q_type}: {count} questions ({percentage:.1f}%)")
            
            print("\nDifficulty distribution:")
            for difficulty, count in overall_difficulty_counts.items():
                percentage = (count / len(all_questions)) * 100
                print(f"  - {difficulty}: {count} questions ({percentage:.1f}%)")
            
            print("\nValidation results:")
            validation_rate = (valid_queries / len(all_questions)) * 100 if all_questions else 0
            print(f"  - Valid queries: {valid_queries} ({validation_rate:.1f}%)")
            print(f"  - Invalid queries: {invalid_queries}")
                
            print(f"\nJSON file saved successfully at: {output_file}")
            print(f"Summary saved at: {summary_file}")
        else:
            print("Warning: No questions were generated. The output file will be empty.")
            # Create an empty file for debugging purposes
            output_file = os.path.join(output_dir, f"{llm_choice}_generated_sample3.json")
            with open(output_file, "w") as f:
                json.dump([], f)
            print(f"Empty JSON file created at: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        raise e

def parse_args():
    parser = argparse.ArgumentParser(description='Generate SQL samples using different LLMs')
    parser.add_argument('--llm', type=str, required=True, choices=['claude', 'openai', 'groq'],
                        help='LLM to use for generation (claude, openai, or groq)')
    parser.add_argument('--samples', type=int, default=25,
                        help='Number of samples to generate per database (default: 25)')
    parser.add_argument('--output', type=str, default='./sample',
                        help='Output directory path (default: ./sample)')
    parser.add_argument('--balanced', action='store_true',
                        help='Ensure a balanced distribution of question types and difficulty levels')
    parser.add_argument('--types', type=str, default='all',
                        help='Comma-separated list of question types to generate (default: all). Options: exact,aggregation,comparison,ranking,reasoning,multi-table,nested')
    parser.add_argument('--difficulties', type=str, default='all',
                        help='Comma-separated list of difficulty levels to generate (default: all). Options: simple,moderate,challenge')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Ensure output directory is absolute
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(os.path.join(CURRENT_DIR, args.output))
    
    # Parse question types
    if args.types.lower() == 'all':
        selected_types = ["exact", "aggregation", "comparison", "ranking", "reasoning", "multi-table", "nested"]
    else:
        selected_types = [t.strip() for t in args.types.split(',')]
        # Validate selected types
        valid_types = ["exact", "aggregation", "comparison", "ranking", "reasoning", "multi-table", "nested"]
        for t in selected_types:
            if t not in valid_types:
                print(f"Warning: Invalid question type '{t}'. Will be ignored.")
                selected_types.remove(d)
    
    # Parse difficulty levels
    if args.difficulties.lower() == 'all':
        selected_difficulties = ["simple", "moderate", "challenge"]
    else:
        selected_difficulties = [d.strip() for d in args.difficulties.split(',')]
        # Validate difficulty levels
        valid_difficulties = ["simple", "moderate", "challenge"]
        for d in selected_difficulties:
            if d not in valid_difficulties:
                print(f"Warning: Invalid difficulty level '{d}'. Will be ignored.")
                selected_difficulties.remove(d)
    
    print(f"Generating samples using {args.llm.upper()}")
    print(f"Number of samples per database: {args.samples}")
    print(f"Output directory: {args.output}")
    print(f"Balanced distribution: {args.balanced}")
    print(f"Question types: {', '.join(selected_types)}")
    print(f"Difficulty levels: {', '.join(selected_difficulties)}")
    
    # Include filter criteria in the output filename
    filename_suffix = ""
    if args.types != 'all':
        filename_suffix += f"_{'-'.join(selected_types)}"
    if args.difficulties != 'all':
        filename_suffix += f"_{'-'.join(selected_difficulties)}"
    if args.balanced:
        filename_suffix += "_balanced"
    
    # Update output directory to include the selection criteria
    if filename_suffix:
        args.output = os.path.join(args.output, f"{args.llm}{filename_suffix}")
    
    generate_sample(args.llm, args.samples, args.output)
