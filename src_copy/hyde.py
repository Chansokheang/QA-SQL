import os
import logging
from openai import OpenAI
import anthropic
from langchain_groq import ChatGroq

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from src.utils import extract_sql

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Get configuration from environment variables or use defaults
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(CURRENT_DIR, "..", "database")
CHROMA_DIR = os.path.join(DATABASE_DIR, "chroma")

# Allow override of persistent directory via environment variable
PERSISTENT_DIR = os.environ.get(
    "RAG2SQL_PERSISTENT_DIR", 
    os.path.join(CHROMA_DIR, "chroma_db")
)

# Get LLM choice from environment variable or default to groq
LLM_CHOICE = os.environ.get("RAG2SQL_LLM_CHOICE", "groq")

# Initialize clients
openai_client = OpenAI()
claude_client = anthropic.Anthropic()
groq_llm = ChatGroq(
    model="Llama-3.3-70b-Versatile",
    temperature=0.7,
    groq_api_key="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"
)

embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

# Creating the Prompt Template for hypothetical document generation
hyde_template = """For the given question about a database schema and SQL queries, generate a hypothetical document that would be helpful for answering this question.
This document should describe the schema structure, relationships between tables, and examples of SQL queries that might be relevant.
Make the document detailed and informative, as if it were written by a database expert.

Only generate the hypothetical document and nothing else:

Question: {question}
"""

hyde_prompt = ChatPromptTemplate.from_template(hyde_template)

def generate_hypothetical_document(question, llm_choice="openai"):
    """
    Generate a hypothetical document for the given question using the specified LLM.
    
    Args:
        question (str): User's question about database/SQL
        llm_choice (str): The LLM to use (openai, claude, or groq)
    
    Returns:
        str: Generated hypothetical document
    """
    query = hyde_prompt.format(question=question)
    
    logging.info(f"Generating hypothetical document using {llm_choice} LLM...")
    
    try:
        if llm_choice == "openai":
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": query}],
                max_tokens=1024,
                temperature=0
            )
            return response.choices[0].message.content
        
        elif llm_choice == "claude":
            response = claude_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                temperature=0,
                max_tokens=1000,
                messages=[{"role": "user", "content": query}]
            )
            return response.content[0].text
        
        elif llm_choice == "groq":
            messages = [{"role": "user", "content": query}]
            response = groq_llm.invoke(messages)
            
            if hasattr(response, "content"):
                return response.content.strip()
            else:
                raise ValueError("LLM response does not contain a valid 'content' attribute.")
        
        else:
            raise ValueError(f"Unsupported LLM choice: {llm_choice}. Must be 'openai', 'claude', or 'groq'.")
    
    except Exception as e:
        logging.error(f"Error generating hypothetical document: {e}", exc_info=True)
        # Return a simple default in case of error
        return f"SQL query for {question}"

def get_hyde_retriever(vector_store, llm_choice="openai"):
    """
    Create a retriever that uses HyDE (Hypothetical Document Embeddings) approach.
    
    Args:
        vector_store: The vector store to retrieve documents from
        llm_choice: The LLM to use for generating hypothetical documents
    
    Returns:
        callable: A function that takes a question and returns retrieved documents
    """
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3},
    )
    
    def hyde_retrieve(question):
        # Generate hypothetical document
        hypothetical_doc = generate_hypothetical_document(question, llm_choice)
        logging.info(f"Generated hypothetical document: {hypothetical_doc[:100]}...")
        
        # Use hypothetical document for retrieval instead of original question
        return base_retriever.invoke(hypothetical_doc)
    
    return hyde_retrieve

