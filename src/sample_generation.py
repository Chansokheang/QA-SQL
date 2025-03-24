import os
import json
import argparse
import anthropic
from openai import OpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from utils import groq_extract_content, claude_extract_content, ollama_extract_content, openai_extract_content, create_persistent_db, extract_evidence
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
        
        # Ensure DB_DIR is a directory
        if not os.path.isdir(DB_DIR):
            raise NotADirectoryError(f"DB_DIR is not a directory: {DB_DIR}")

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
                - Ensure the output is in JSON format with the following fields:
                    - `question_id`: A unique numeric ID for each question.
                    - `db_id`: The name of the database (e.g., "{db_name}").
                    - `question`: The question in natural language.
                    - `evidence`: Use the provided evidence to justify the SQL query.
                    - `SQL`: A valid SQL query based on the schema and evidence and wrap table name like `order`, `bond`,... .
                    - `difficulty`: A difficulty level for the question (e.g., "simple", "moderate", "challenge").

                **Example Output:**
                [
                    {{
                        "question_id": 0,
                        "db_id": "{db_name}",
                        "question": "What is the total amount of money transferred from account 1 to account 2?",
                        "evidence": "Transferred amount = SUM(`amount`) WHERE `account_id` = 1 AND `account_to` = 2",
                        "SQL": "SELECT SUM(amount) FROM transactions WHERE account_id = 1 AND account_to = 2;",
                        "difficulty": "simple"
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
                for question in json_data:
                    question["question_id"] = question_id_counter  # Assign unique ID inside each object
                    question_id_counter += 1  # Increment counter
                    all_questions.append(question)
            else:
                print(f"Warning: No valid data returned for database {db_name}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Check if we have any data before writing the file
        if all_questions:
            output_file = os.path.join(output_dir, f"{llm_choice}_generated_sample.json")
            with open(output_file, "w") as f:
                json.dump(all_questions, f, indent=4)
            print(f"JSON file saved successfully at: {output_file} with {len(all_questions)} questions")
        else:
            print("Warning: No questions were generated. The output file will be empty.")
            # Create an empty file for debugging purposes
            output_file = os.path.join(output_dir, f"{llm_choice}_generated_sample.json")
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Ensure output directory is absolute
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(os.path.join(CURRENT_DIR, args.output))
    
    print(f"Generating samples using {args.llm.upper()}")
    print(f"Number of samples per database: {args.samples}")
    print(f"Output directory: {args.output}")
    
    generate_sample(args.llm, args.samples, args.output)
