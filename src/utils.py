import re
import logging
import sqlite3
import os
import json
import anthropic

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from openai import OpenAI
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)

claude_client = anthropic.Anthropic()

llm = ChatGroq(
    model="Llama-3.3-70b-Versatile",
    temperature=0.7,
)

embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database/dev_folder/dev_databases"))
DEV_FOLDER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database/dev_folder"))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(CURRENT_DIR, "evidence")
DATABASE_DIR = os.path.join(CURRENT_DIR, "..", "database")
CHROMA_DIR = os.path.join(DATABASE_DIR, "chroma")
EVIDENCE_DIR = os.path.join(CHROMA_DIR, "evidence_chroma_db")
PERSISTENT_DIR = os.path.join(CHROMA_DIR, "chroma_db")

def extract_evidence():
    # Ensure DB_DIR is a directory
    if not os.path.isdir(DB_DIR):
        raise NotADirectoryError(f"DB_DIR is not a directory: {DEV_FOLDER_DIR}")
    
    dev_path = os.path.join(DEV_FOLDER_DIR, "dev.json")
    with open(dev_path, 'r') as f:
        data = json.load(f)
    
    evidence_arr = []

    # Extract db_id and evidence
    grouped_evidence = {}
    for item in data:
        db_id = item["db_id"]
        evidence = item["evidence"]
        
        # Group evidence by db_id
        if db_id not in grouped_evidence:
            grouped_evidence[db_id] = []
        grouped_evidence[db_id].append(evidence)

    # Print grouped evidence by db_id
    for db_id, evidences in grouped_evidence.items():
        for evidence in evidences:
            evidence_arr.append(evidence)
        with open(f"./newEvidence/{db_id}_evidence.json", "w") as f:
            json.dump(evidence_arr, f, indent=4)
        evidence_arr.clear()

def create_persistent_db():
    all_documents = []
    # Step 1: Iterate through all JSON files in the directory
    for file_name in os.listdir(FILE_DIR):
        if file_name.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(FILE_DIR, file_name)
            print(f"Processing file: {file_path}")

            # Step 2: Load JSON content (list of strings)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Step 3: Convert each string to a Document object
            for content in data:
                doc = Document(page_content=content, metadata={"source": file_name})
                all_documents.append(doc)

    # Step 4: Embed all documents into a Chroma vector store
    vector_store = Chroma.from_documents(
        documents=all_documents,
        collection_name="evidence_collection",
        persist_directory=EVIDENCE_DIR,
        embedding=embedding_function
    )
    print(f"✅ Stored {len(all_documents)} documents in ChromaDB at {PERSISTENT_DIR}")
    return vector_store
        

def extract_sql (response: str) -> str:
    """
    Extracts SQL code from a response string.
    
    Args:
        response (str): The response containing SQL code.
    
    Returns:
        str: Extracted SQL code or the original response if parsing fails.
    """
    try:
        match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
        if match:
            sql = match.group(1).strip()
            return sql
        else:
            logging.warning("No SQL code block found in the response. Returning the original response.")
            return response.strip()
    except Exception as e:
        logging.error(f"Error occurred while extracting SQL: {e}", exc_info=True)
        raise
def get_relevant_schemas(question: str, db_name: str):
    import os
    try:
        "Get the database path and schema information"
        db_path = f'./database/dev_folder/dev_databases/{db_name}/{db_name}.sqlite'
        db_path = os.path.abspath(db_path)
        if not os.path.exists(db_path):
            print(f"Error: Database file does not exist at path {db_path}")
        engine = create_engine(f'sqlite:///{db_path}')
        db = SQLDatabase(engine)
        schemas = db.get_table_info()
        
        return schemas
    except Exception as e:
        # print("error ",e)
        raise
    

def generate_sql(contextual, schema: str, question: str, llm_choice="groq"):
    """
    Generate SQL based on the given context, schema, and question using the specified LLM.
    
    Args:
        contextual: Contextual information from vector store
        schema: Database schema information
        question: User's question
        llm_choice: The LLM to use (openai, claude, or groq)
    
    Returns:
        str: Generated SQL query
    """
    messages = [
        {
            "role" : "assistant",
            "content" : (
                f"""
                    You are an expert SQL generator. Your task is to generate SQL queries based on the provided database schema, contextual information, and the user's question. Use the context and schema carefully to understand the data structure and answer the question accurately with SQL.

                    **Relevant Schema**:
                    {schema}
            
                    **Contextual Information**:
                    {contextual}
            
                    **IMPORTANT**:
                        - First, determine if the question matches the context provided or is thematically similar. If it does, generate SQL that leverages this context directly.
                        - Enclose columns with spaces or special characters in double quotes.
                        - If the question does not match the context, use the schema and context as a guide to understand the data structure and generate SQL that best answers the user's question.
                        - Only use columns from the schema. Do not introduce columns not listed here.


                    **SQL Code**:
                    (Provide only the SQL code, without any additional explanations or comments and no additional messages.)
                """)
        },
        {
            "role" : "user",
            "content" : (f"Generate a SQL query for: {question}")
        }
    ]
    
    print(f"Generating SQL using {llm_choice} LLM...")
    
    if llm_choice == "openai":
        response = OpenAI().chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024,
            temperature=0
        )
        sql = extract_sql(response.choices[0].message.content)
    
    elif llm_choice == "claude":
        response = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            temperature=0,
            max_tokens=4000,
            messages=messages
        )
        sql = extract_sql(response.content[0].text)
    
    elif llm_choice == "groq":
        # Invoke Groq's Llama3.3 model
        response = llm.invoke(messages)
        
        # Ensure response is an AIMessage and extract the content
        if hasattr(response, "content"):
            raw_sql = response.content.strip()  # Extract content correctly
        else:
            raise ValueError("LLM response does not contain a valid 'content' attribute.")
            
        sql = extract_sql(raw_sql)
    
    else:
        raise ValueError(f"Unsupported LLM choice: {llm_choice}. Must be 'openai', 'claude', or 'groq'.")
    
    return sql


def execute_sql(sql_query: str, db_name: str):
    try:
        conn = sqlite3.connect(f'../database/dev_folder/dev_databases/{db_name}/{db_name}.sqlite')
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
        return result
    except sqlite3.Error as e:
        logging.error(f"SQL Execution ERROR: {e}", exc_info=True)
        return None


# datasets is dev.json
def decouple_question(datasets):
    question_list = []
    db_name_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        db_name_list.append(data["db_id"])

    return question_list, db_name_list


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def generate_sql_file(sql_lst, output_path=None, append=False):
    """
    Function to save the SQL results to a file.
    
    Args:
        sql_lst (list): List of SQL strings to save (with BIRD format)
        output_path (str): Path to save the results
        append (bool): Whether to append to existing file (default: False)
    """
    # If appending and file exists, load existing data first
    result = {}
    start_idx = 0
    
    if append and output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            
            # Filter to keep only entries with BIRD delimiter
            result = {k: v for k, v in existing_data.items() if '\t----- bird -----\t' in v}
            
            # Find the highest existing key to determine where to start numbering
            if result:
                # Convert string keys to integers for proper comparison
                int_keys = [int(k) for k in result.keys() if k.isdigit()]
                if int_keys:
                    start_idx = max(int_keys) + 1
            
            print(f"Loaded {len(result)} valid entries from {output_path}, continuing from index {start_idx}")
            if len(existing_data) != len(result):
                print(f"Removed {len(existing_data) - len(result)} entries without BIRD format")
        except Exception as e:
            print(f"Error loading existing file for append: {e}")
            print("Starting with empty results")
    
    # Create a new result dictionary with proper indexing
    # Ensure all entries have the BIRD format
    for i, sql in enumerate(sql_lst):
        if '\t----- bird -----\t' in sql:
            result[str(start_idx + i)] = sql
        else:
            print(f"Warning: Skipping entry without BIRD format at index {start_idx + i}")
    
    # Save to file if output_path is provided
    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)  # Ensure the directory exists
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)

    return result


def groq_extract_content(response):
    raw_content = response.content.strip()
    all_json_data = []
    import re
    match = re.search(r"(\[.*\])", raw_content, re.DOTALL)  # Match JSON array

    if match:
        json_data = json.loads(match.group(1))  # Extract and parse JSON
    else:
        raise ValueError("No valid JSON found in the LLM response.")
            
    for i, item in enumerate(json_data):
        item["question_id"] = len(all_json_data) + i  # Ensure unique question IDs
    all_json_data.extend(json_data)
    return all_json_data

def openai_extract_content(response): 
    if isinstance(response, str):  
        raw_content = response.strip()  # If response is a string, use it directly
    elif hasattr(response, 'choices') and response.choices:  
        raw_content = response.choices[0].message.content.strip()  # Extract OpenAI API response
    else:
        raise ValueError("Invalid response format. Expected a string or OpenAI API response object.")

    all_json_data = []

    # Match JSON array inside response
    match = re.search(r"(\[.*\])", raw_content, re.DOTALL)

    if match:
        json_data = json.loads(match.group(1))  # Extract and parse JSON
    else:
        raise ValueError("No valid JSON found in the LLM response.")
    
    # Ensure unique question IDs
    for i, item in enumerate(json_data):
        item["question_id"] = len(all_json_data) + i  
    
    all_json_data.extend(json_data)
    return all_json_data

def ollama_extract_content(response):
    if not isinstance(response, str):  # Ensure response is a string
        raise ValueError("Expected response to be a string, but got a different type.")

    raw_content = response.strip()  # Remove leading/trailing spaces
    all_json_data = []
    
    # Match JSON array inside response
    match = re.search(r"(\[.*\])", raw_content, re.DOTALL)

    if match:
        json_data = json.loads(match.group(1))  # Extract and parse JSON
    else:
        raise ValueError("No valid JSON found in the LLM response.")
    
    # Ensure unique question IDs
    for i, item in enumerate(json_data):
        item["question_id"] = len(all_json_data) + i
    
    all_json_data.extend(json_data)
    return all_json_data

def claude_extract_content(response):
    """
    Extracts JSON content from Claude's response.
    Returns an empty list if no valid JSON is found.
    """
    import re
    print(f"Processing Claude response: {response[:100]}...") # Print first 100 chars for debugging
    
    # Extract JSON using regex (assuming JSON is enclosed in square brackets [])
    match = re.search(r'(\[.*\])', response, re.DOTALL)

    if match:
        raw_json = match.group(1)  # Extract JSON string
        print(f"Found JSON match of length: {len(raw_json)}")
                
        # Remove invalid control characters (like newlines, tabs)
        raw_json = raw_json.replace("\n", " ").replace("\t", " ").strip()

        # Load JSON safely
        try:
            json_data = json.loads(raw_json)
            print(f"Successfully parsed JSON with {len(json_data)} items")
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw JSON response (first 200 chars): {raw_json[:200]}")
            return []
    else:
        print("No JSON found in response")
        # Try a more lenient approach to find any JSON-like structure
        try:
            # Look for content that might be JSON but not in brackets
            possible_json = re.search(r'(\{.*\})', response, re.DOTALL)
            if possible_json:
                # Try wrapping it in brackets to make it a list
                bracketed_json = f"[{possible_json.group(1)}]"
                json_data = json.loads(bracketed_json)
                print(f"Found and parsed JSON object as list")
                return json_data
        except:
            pass
            
        return []
