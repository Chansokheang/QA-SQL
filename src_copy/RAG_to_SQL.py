import os
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field

from src.utils import get_relevant_schemas, generate_sql
from src.hyde import get_hyde_retriever

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
    os.path.join(CHROMA_DIR, "unify_chroma3")
)

# Get LLM choice from environment variable or default to groq
LLM_CHOICE = os.environ.get("RAG2SQL_LLM_CHOICE", "groq")

# Get HyDE mode from environment variables (default to True)
USE_HYDE = os.environ.get("RAG2SQL_USE_HYDE", "true").lower() == "true"

print(f"RAG_to_SQL initialized with:")
print(f"  - Persistent directory: {PERSISTENT_DIR}")
print(f"  - LLM choice: {LLM_CHOICE}")
print(f"  - Using HyDE: {USE_HYDE}")

embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = Chroma(
    collection_name="generated_queries_collection",
    persist_directory=PERSISTENT_DIR,
    embedding_function=embedding_function
)

# Set up the retriever based on whether we use HyDE or not
if USE_HYDE:
    # Use HyDE-based retriever
    retriever = get_hyde_retriever(vector_store, LLM_CHOICE)
else:
    # Use standard retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3},
    )

chain = (
    RunnableParallel(
        branches = {
            "schema_retriever" : RunnableLambda(lambda x: get_relevant_schemas(x['question'], x['db_name'])),
            "context_retriever" : lambda x: retriever(x['question']) if USE_HYDE else retriever.invoke(x['question']),
            "input" : lambda x: x['question'],
            "llm_choice" : lambda x: LLM_CHOICE  # Pass the LLM choice to generate_sql
        }
    )
    | RunnableLambda (
        lambda x: generate_sql(
            x['branches']['context_retriever'],
            x['branches']['schema_retriever'],
            x['branches']['input'],
            llm_choice=x['branches']['llm_choice']
        )
    )
)


class Request(BaseModel):
    input: str = Field(
        ...
    )
    db_name: str = Field(
        ...
    )
    
chain = chain.with_types(input_type=Request)