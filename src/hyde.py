import os
import logging
from openai import OpenAI
import anthropic
from langchain_groq import ChatGroq

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document # Needed for type hinting

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
    os.path.join(CHROMA_DIR, "chroma_db0304")
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


def get_hyde_retriever(vector_store, llm_choice="openai", top_n=5):
    """
    Create a retriever that uses a hybrid HyDE approach followed by Cohere Rerank.
    Retrieves based on both the hypothetical document and the original question,
    then reranks the combined results using Cohere Rerank.

    Args:
        vector_store: The vector store to retrieve documents from.
        llm_choice: The LLM to use for generating hypothetical documents.
        top_n (int): The final number of documents to return after reranking.

    Returns:
        callable: A function that takes a question and returns reranked documents.
    """
    # Base retriever for initial similarity search
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        # Retrieve more initially to give the reranker more options
        search_kwargs={"k": 10, "score_threshold": 0.2}, 
    )

    # Reranker setup (assumes COHERE_API_KEY is in environment)
    try:
        compressor = CohereRerank(top_n=top_n)
        logging.info("Cohere Rerank compressor initialized.")
        use_reranker = True
    except Exception as e:
        logging.error(f"Failed to initialize Cohere Rerank. Ensure COHERE_API_KEY is set. Falling back to base retriever without reranking. Error: {e}")
        # Fallback: Don't use the reranker if initialization fails
        use_reranker = False
        compressor = None # Ensure compressor is None if not used

    def hyde_hybrid_rerank_retrieve(question: str) -> list[Document]:
        # 1. Retrieve based on the original question
        logging.info(f"Retrieving based on original question: {question[:100]}...")
        original_results = base_retriever.invoke(question)
        logging.info(f"Found {len(original_results)} results based on original question.")

        # 2. Generate hypothetical document and retrieve based on it
        hypothetical_doc = generate_hypothetical_document(question, llm_choice)
        logging.info(f"Generated hypothetical document: {hypothetical_doc[:100]}...")
        hyde_results = base_retriever.invoke(hypothetical_doc)
        logging.info(f"Found {len(hyde_results)} results based on hypothetical document.")

        # 3. Combine and deduplicate results
        combined_results_dict = {doc.page_content: doc for doc in original_results}
        for doc in hyde_results:
            if doc.page_content not in combined_results_dict:
                combined_results_dict[doc.page_content] = doc
        
        initial_combined_results = list(combined_results_dict.values())
        logging.info(f"Combined and deduplicated initial results: {len(initial_combined_results)} documents.")

        if not initial_combined_results:
             return [] # Return empty list if no initial results

        # 4. Rerank the combined results using the original question IF reranker is available
        if use_reranker and compressor:
            logging.info(f"Reranking {len(initial_combined_results)} documents with Cohere Rerank...")
            reranked_docs = compressor.compress_documents(
                documents=initial_combined_results, 
                query=question
            )
            logging.info(f"Reranked results: {len(reranked_docs)} documents.")
            return reranked_docs
        else:
            # Fallback: Return the combined results without reranking, potentially truncated
            logging.warning("Cohere Reranker not available or failed. Returning top results from initial retrieval.")
            # Return top_n results based on initial retrieval (no specific order guaranteed here without scores)
            return initial_combined_results[:top_n] 

    return hyde_hybrid_rerank_retrieve

def get_reranked_retriever(vector_store, top_n=5):
    """
    Create a retriever that uses Cohere Rerank on the standard retriever.

    Args:
        vector_store: The vector store to retrieve documents from.
        top_n (int): The final number of documents to return after reranking.

    Returns:
        callable: A function that takes a question and returns reranked documents.
    """
    # Base retriever for initial similarity search
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.3},
    )

    # Reranker setup (assumes COHERE_API_KEY is in environment)
    try:
        compressor = CohereRerank(top_n=top_n)
        logging.info("Cohere Rerank compressor initialized.")
        use_reranker = True
    except Exception as e:
        logging.error(f"Failed to initialize Cohere Rerank. Ensure COHERE_API_KEY is set. Falling back to base retriever without reranking. Error: {e}")
        # Fallback: Don't use the reranker if initialization fails
        use_reranker = False
        compressor = None  # Ensure compressor is None if not used

    def rerank_retrieve(question: str) -> list[Document]:
        # 1. Retrieve based on the original question
        logging.info(f"Retrieving based on original question: {question[:100]}...")
        initial_results = base_retriever.invoke(question)
        logging.info(f"Found {len(initial_results)} results based on original question.")

        if not initial_results:
            return []  # Return empty list if no initial results

        # 2. Rerank the initial results using the original question IF reranker is available
        if use_reranker and compressor:
            logging.info(f"Reranking {len(initial_results)} documents with Cohere Rerank...")
            reranked_docs = compressor.compress_documents(
                documents=initial_results,
                query=question
            )
            logging.info(f"Reranked results: {len(reranked_docs)} documents.")
            return reranked_docs
        else:
            # Fallback: Return the initial results without reranking, potentially truncated
            logging.warning("Cohere Reranker not available or failed. Returning top results from initial retrieval.")
            return initial_results[:top_n]

    return rerank_retrieve
