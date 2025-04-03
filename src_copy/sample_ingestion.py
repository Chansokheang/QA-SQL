import os
import argparse

from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def ingest_dataset(dataset_path, persistent_dir):
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

    loader = JSONLoader(
        file_path=dataset_path,
        jq_schema='.[] | {page_content: .question, metadata: {query: .SQL}}',
        text_content=False
    )

    documents = loader.load()

    vector_store = Chroma.from_documents(
        documents=documents,
        collection_name="generated_queries_collection",
        persist_directory=persistent_dir,
        embedding=embedding_function
    )
    
    print(f"Successfully ingested {len(documents)} documents from {dataset_path}")
    print(f"Vector store persisted to {persistent_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest dataset into Chroma vector store")
    parser.add_argument("--dataset", type=str, help="Path to the dataset JSON file")
    parser.add_argument("--persistent_dir", type=str, help="Directory to persist Chroma DB")
    
    args = parser.parse_args()
    
    # Use provided arguments or default values
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    dataset_path = args.dataset
    if not dataset_path:
        dataset_path = os.path.join(CURRENT_DIR, "sample", "Groq_generated_sample.json")
        print(f"Using default dataset path: {dataset_path}")
    
    persistent_dir = args.persistent_dir
    if not persistent_dir:
        DATABASE_DIR = os.path.join(CURRENT_DIR, "..", "database")
        CHROMA_DIR = os.path.join(DATABASE_DIR, "chroma")
        persistent_dir = os.path.join(CHROMA_DIR, "chroma_db")
        print(f"Using default persistent directory: {persistent_dir}")
    
    # Create directories if they don't exist
    os.makedirs(persistent_dir, exist_ok=True)
    
    ingest_dataset(dataset_path, persistent_dir)