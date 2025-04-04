import re
import logging
import sqlite3
import os
import json
import anthropic
from typing import List, Dict, Any, Optional

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from openai import OpenAI
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:
    logging.warning("sentence-transformers not installed. Enhanced retrieval will not be available unless package is installed.")
    CrossEncoder = None
    SentenceTransformer = None

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)

claude_client = anthropic.Anthropic()

llm = ChatGroq(
    model="Llama-3.3-70b-Versatile",
    temperature=0.7,
    groq_api_key="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"
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

class HybridRetriever(BaseRetriever):
    """A retriever that combines vector similarity search with keyword search."""
    
    def __init__(
        self, 
        vector_store: Chroma, 
        k: int = 5, 
        score_threshold: float = 0.3,
        keyword_weight: float = 0.3
    ):
        """Initialize the hybrid retriever.
        
        Args:
            vector_store: The vector store to use for similarity search
            k: Number of documents to retrieve (default: 5)
            score_threshold: Minimum similarity score (default: 0.3)
            keyword_weight: Weight for keyword search vs vector search (default: 0.3)
        """
        super().__init__()
        # Store all parameters as private attributes
        self._vector_store = vector_store
        self._k = k
        self._score_threshold = score_threshold
        self._keyword_weight = keyword_weight

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query using both vector and keyword search.
        
        Args:
            query: The query to search for
            run_manager: The callback manager
            
        Returns:
            List of relevant documents
        """
        # Vector search
        vector_docs = self._vector_store.similarity_search_with_score(
            query=query, k=self._k
        )
        
        # Add vector score to metadata
        for doc, score in vector_docs:
            doc.metadata["vector_score"] = score
            doc.metadata["keyword_score"] = 0.0
            doc.metadata["combined_score"] = (1 - self._keyword_weight) * score
        
        # Keyword search (simple implementation)
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Get all documents from the collection for keyword search
        all_docs = self._vector_store.get()


class EnhancedHybridRetriever(BaseRetriever):
    """An enhanced retriever that combines vector search, keyword search, and cross-encoder reranking."""
    
    def __init__(
        self, 
        vector_store: Chroma, 
        k: int = 5, 
        rerank_k: int = 10,  # Retrieve more initially to rerank
        score_threshold: float = 0.3,
        keyword_weight: float = 0.3,
        cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ):
        """Initialize the enhanced hybrid retriever.
        
        Args:
            vector_store: The vector store to use for similarity search
            k: Number of final documents to retrieve (default: 5)
            rerank_k: Number of documents to retrieve initially for reranking (default: 10)
            score_threshold: Minimum similarity score (default: 0.3)
            keyword_weight: Weight for keyword search vs vector search (default: 0.3)
            cross_encoder_model: The cross-encoder model to use for reranking (default: 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        """
        super().__init__()
        self._vector_store = vector_store
        self._k = k
        self._rerank_k = rerank_k
        self._score_threshold = score_threshold
        self._keyword_weight = keyword_weight
        
        # Initialize cross-encoder for reranking
        if CrossEncoder is not None:
            try:
                self._cross_encoder = CrossEncoder(cross_encoder_model)
                logging.info(f"Initialized cross-encoder model: {cross_encoder_model}")
            except Exception as e:
                logging.error(f"Failed to initialize cross-encoder: {e}")
                self._cross_encoder = None
        else:
            logging.warning("CrossEncoder not available. Install sentence-transformers for reranking.")
            self._cross_encoder = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query using enhanced hybrid retrieval.
        
        Args:
            query: The query to search for
            run_manager: The callback manager
            
        Returns:
            List of reranked relevant documents
        """
        # Vector search - get more documents initially for reranking
        vector_docs = self._vector_store.similarity_search_with_score(
            query=query, k=self._rerank_k
        )
        
        # Add vector score to metadata
        for doc, score in vector_docs:
            doc.metadata["vector_score"] = score
            doc.metadata["keyword_score"] = 0.0
            doc.metadata["combined_score"] = (1 - self._keyword_weight) * score
        
        # Keyword search (simple implementation)
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Get all documents from the collection for keyword search
        all_docs = self._vector_store.get()
        
        # Sort docs with vector scores by vector score
        vector_docs_dict = {doc.page_content: (doc, score) for doc, score in vector_docs}
        
        # For each document, calculate keyword match score (similar to HybridRetriever)
        keyword_results = []
        for i, doc_content in enumerate(all_docs["documents"]):
            if not doc_content:  # Skip empty documents
                continue
                
            doc_text = doc_content.lower()
            doc_keywords = set(re.findall(r'\b\w+\b', doc_text))
            
            # Calculate keyword match score (Jaccard similarity)
            if not query_keywords or not doc_keywords:
                keyword_score = 0.0
            else:
                intersection = len(query_keywords.intersection(doc_keywords))
                union = len(query_keywords.union(doc_keywords))
                keyword_score = intersection / union if union > 0 else 0.0
            
            # Skip documents with low keyword score
            if keyword_score < 0.1:  # Minimum threshold for keyword matches
                continue
                
            # Create document with metadata
            metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
            metadata["keyword_score"] = keyword_score
            
            # Check if this document is already in vector results
            if doc_content in vector_docs_dict:
                # Update the existing document with keyword score
                existing_doc, vector_score = vector_docs_dict[doc_content]
                existing_doc.metadata["keyword_score"] = keyword_score
                existing_doc.metadata["combined_score"] = (
                    (1 - self._keyword_weight) * vector_score + 
                    self._keyword_weight * keyword_score
                )
            else:
                # New document from keyword search
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        **metadata,
                        "vector_score": 0.0,
                        "combined_score": self._keyword_weight * keyword_score
                    }
                )
                keyword_results.append(doc)
        
        # Combine results from vector search and keyword search
        all_results = [doc for doc, _ in vector_docs] + keyword_results
        
        # Apply cross-encoder reranking if available
        if self._cross_encoder is not None and len(all_results) > 0:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in all_results]
            
            # Get cross-encoder scores
            try:
                cross_scores = self._cross_encoder.predict(pairs)
                
                # Add cross-encoder scores to metadata
                for i, doc in enumerate(all_results):
                    doc.metadata["cross_score"] = float(cross_scores[i])
                    # Final score is a weighted combination
                    doc.metadata["final_score"] = doc.metadata.get("combined_score", 0) * 0.3 + float(cross_scores[i]) * 0.7
                
                # Sort by final score
                all_results.sort(key=lambda x: x.metadata.get("final_score", 0), reverse=True)
                logging.info(f"Applied cross-encoder reranking to {len(all_results)} documents")
            except Exception as e:
                logging.error(f"Error in cross-encoder reranking: {e}")
                # Fall back to combined scores if reranking fails
                all_results.sort(key=lambda x: x.metadata.get("combined_score", 0), reverse=True)
        else:
            # Sort by combined score if no cross-encoder
            all_results.sort(key=lambda x: x.metadata.get("combined_score", 0), reverse=True)
        
        # Return top k results
        return all_results[:self._k]

def decompose_complex_query(query, db_schema, llm_choice="groq"):
    """Break down complex queries into simpler sub-queries.
    
    Args:
        query: The original complex query
        db_schema: Database schema information to help with decomposition
        llm_choice: The LLM to use (openai, claude, or groq)
        
    Returns:
        List of sub-queries
    """
    decomposition_prompt = f"""
    I need to break down a complex database question into simpler sub-questions.
    
    Database schema:
    {db_schema}
    
    Complex question: {query}
    
    Please break this down into 2-3 simpler sub-questions that would help answer the original question.
    Format: JSON array of strings, each string being a simpler sub-question.
    """
    
    messages = [
        {
            "role": "user",
            "content": decomposition_prompt
        }
    ]
    
    # Get response from LLM
    try:
        if llm_choice == "openai":
            response = OpenAI().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1024,
                temperature=0
            )
            response_text = response.choices[0].message.content
        
        elif llm_choice == "claude":
            response = claude_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                temperature=0,
                max_tokens=4000,
                messages=messages
            )
            response_text = response.content[0].text
        
        elif llm_choice == "groq":
            response = llm.invoke(messages)
            if hasattr(response, "content"):
                response_text = response.content.strip()
            else:
                logging.error("LLM response does not contain a valid 'content' attribute.")
                return [query]  # Return original query if decomposition fails
        else:
            logging.error(f"Unsupported LLM choice: {llm_choice}")
            return [query]  # Return original query if LLM choice is invalid
        
        # Try to parse as JSON
        try:
            sub_queries = json.loads(response_text)
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                logging.info(f"Successfully decomposed query into {len(sub_queries)} sub-queries")
                return sub_queries
        except json.JSONDecodeError:
            # Fall back to regex if JSON parsing fails
            sub_queries = re.findall(r'"([^"]+)"', response_text)
            if sub_queries:
                logging.info(f"Extracted {len(sub_queries)} sub-queries using regex")
                return sub_queries
        
        # If decomposition failed, return original query
        logging.warning("Query decomposition failed, returning original query")
        return [query]
    
    except Exception as e:
        logging.error(f"Error in query decomposition: {e}")
        return [query]  # Return original query if any error occurs

def get_improved_retrieval(query, db_name, db_schema, vector_store, use_decomposition=True, llm_choice="groq"):
    """Get retrieved examples with query decomposition.
    
    Args:
        query: The original user query
        db_name: Database name
        db_schema: Database schema
        vector_store: Vector store for retrieval
        use_decomposition: Whether to use query decomposition
        llm_choice: LLM to use for decomposition
        
    Returns:
        List of retrieved documents
    """
    # Initialize the enhanced hybrid retriever
    retriever = EnhancedHybridRetriever(vector_store=vector_store, k=5, rerank_k=10)
    
    if not use_decomposition:
        # Simple retrieval without decomposition
        return retriever.get_relevant_documents(query)
    
    # Decompose query
    sub_queries = decompose_complex_query(query, db_schema, llm_choice)
    
    # Get results for each sub-query
    all_docs = []
    for sub_query in sub_queries:
        docs = retriever.get_relevant_documents(sub_query)
        all_docs.extend(docs)
    
    # Remove duplicates
    seen_contents = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
    
    # Re-score based on relevance to original query
    # Simple semantic similarity to original query
    for doc in unique_docs:
        # Use keyword matching for quick similarity
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        doc_keywords = set(re.findall(r'\b\w+\b', doc.page_content.lower()))
        
        if not query_keywords or not doc_keywords:
            relevance = 0.0
        else:
            intersection = len(query_keywords.intersection(doc_keywords))
            union = len(query_keywords.union(doc_keywords))
            relevance = intersection / union if union > 0 else 0.0
        
        doc.metadata["relevance_to_original"] = relevance
    
    # Sort by relevance to original query
    unique_docs.sort(key=lambda x: x.metadata.get("relevance_to_original", 0), reverse=True)
    
    # Take top k results
    return unique_docs[:5]

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
    # Format contextual information with relevance scores if available
    formatted_context = ""
    
    if isinstance(contextual, list) and all(hasattr(doc, 'metadata') for doc in contextual):
        # Sort documents by combined_score if available
        contextual.sort(
            key=lambda x: x.metadata.get("combined_score", 
                 x.metadata.get("vector_score", 0)
            ), 
            reverse=True
        )
        
        # Format each document with its scores
        for i, doc in enumerate(contextual):
            # Get scores from metadata
            vector_score = doc.metadata.get("vector_score", "N/A")
            keyword_score = doc.metadata.get("keyword_score", "N/A")
            combined_score = doc.metadata.get("combined_score", "N/A")
            
            # Format the context with relevance information
            if isinstance(combined_score, float):
                formatted_context += f"\nEXAMPLE {i+1} (Relevance: {combined_score:.4f}):\n{doc.page_content}\n"
            else:
                formatted_context += f"\nEXAMPLE {i+1} (Relevance: {combined_score}):\n{doc.page_content}\n"
    else:
        # Handle the case when contextual is not a list of Documents or is empty
        formatted_context = str(contextual) if contextual else "No relevant examples found."

    messages = [
        {
            "role" : "system", # Use system role for overall instructions
            "content" : (
                f"""You are an expert SQL generator. Your task is to generate a single, correct SQL query based on the provided database schema, example Question/SQL pairs, and the user's question.

                **Relevant Schema**:
                ```sql
                {schema}
                ```

                **Contextual Examples (Question/SQL Pairs)**:
                {formatted_context}

                **Instructions**:
                1. Carefully analyze the user's question: "{question}"
                2. Examine the **Relevant Schema** to understand table structures and relationships.
                3. Review the **Contextual Examples**. Pay close attention to the examples with the highest relevance scores.
                4. **Adapt the SQL pattern** from the most relevant example(s) to answer the user's specific question using the provided schema. If no relevant examples are found, rely solely on the schema.
                5. Ensure the generated SQL query accurately addresses all conditions and requirements mentioned in the user's question.
                6. Only use tables and columns defined in the **Relevant Schema**.
                7. Enclose table or column names with spaces or special characters in double quotes (e.g., "Order Details").
                8. Output **only** the final SQL query within ```sql ... ``` tags. Do not include any other text, explanations, or comments before or after the SQL block.
                """
            )
        },
        {
            "role" : "user",
            "content" : f"Generate the SQL query for the question: \"{question}\""
        }
    ]

    logging.debug(f"Prompt sent to {llm_choice} LLM:\nSystem: {messages[0]['content']}\nUser: {messages[1]['content']}") # Log the prompt
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
        sql_lst (list): List of tuples (index, sql_string) or just sql strings to save (with BIRD format)
        output_path (str): Path to save the results
        append (bool): Whether to append to existing file (default: False)
    """
    # If appending and file exists, load existing data first
    result = {}
    
    if append and output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            
            # Filter to keep only entries with BIRD delimiter
            result = {k: v for k, v in existing_data.items() if '\t----- bird -----\t' in v}
            
            print(f"Loaded {len(result)} valid entries from {output_path}")
            if len(existing_data) != len(result):
                print(f"Removed {len(existing_data) - len(result)} entries without BIRD format")
        except Exception as e:
            print(f"Error loading existing file for append: {e}")
            print("Starting with empty results")
    
    # Add new entries to the result dictionary
    # Check if we have index information in the SQL list (tuple format)
    entries_added = 0
    for item in sql_lst:
        if isinstance(item, tuple) and len(item) == 2:
            # The item is a tuple with (index, sql_content)
            idx, sql = item
            if '\t----- bird -----\t' in sql:
                result[str(idx)] = sql
                entries_added += 1
            else:
                print(f"Warning: Skipping entry without BIRD format at index {idx}")
        else:
            # Legacy format (just the SQL string)
            if '\t----- bird -----\t' in item:
                # Find the next available index if no index was provided
                next_idx = 0
                if result:
                    int_keys = [int(k) for k in result.keys() if k.isdigit()]
                    if int_keys:
                        next_idx = max(int_keys) + 1
                
                result[str(next_idx)] = item
                entries_added += 1
            else:
                print(f"Warning: Skipping entry without BIRD format (no index)")
    
    print(f"Added {entries_added} new entries to BIRD format output")
    
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
