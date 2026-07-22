"""
Test Retrieval Script for RAG System

Tests the Pinecone vector store retrieval using HuggingFace embeddings.
Uses the same embedding model as the data ingestion pipeline (all-MiniLM-L6-v2).

Usage:
    python 1-test-retrieval.py
"""

import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# --- Load and validate environment variables ---
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
if not pinecone_index_name:
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set.")


def create_embeddings(model_name: str = "BAAI/bge-large-en-v1.5") -> HuggingFaceEmbeddings:
    """Create HuggingFace embeddings (free, runs locally).

    Args:
        model_name (str): HuggingFace model name. Must match the model used during ingestion.

    Returns:
        HuggingFaceEmbeddings: Configured embeddings instance.
    """
    print(f"Loading embedding model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model loaded.")
    return embeddings


def search_similar_documents(
    query: str,
    no_of_results: int = 3,
    index_name: str = None,
    embeddings: HuggingFaceEmbeddings = None
) -> list:
    """Search Pinecone for documents similar to the given query.

    Args:
        query (str): The search query text.
        no_of_results (int): Number of results to return. Defaults to 3.
        index_name (str): Pinecone index name. Defaults to env variable.
        embeddings (HuggingFaceEmbeddings): Embeddings instance. Creates one if None.

    Returns:
        list: List of (Document, score) tuples sorted by similarity.

    Raises:
        ValueError: If query is empty.
    """
    if query is None or query.strip() == "":
        raise ValueError("Query must be a non-empty string.")

    if index_name is None:
        index_name = pinecone_index_name

    if embeddings is None:
        embeddings = create_embeddings()

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
    )

    results = vector_store.similarity_search_with_score(query, k=no_of_results)

    return results


if __name__ == "__main__":
    query = """
        Experienced Candidate with Embedded Systems and Firmware Development Skills
        Requirements:
        Bachelor's degree in Computer Science
        At least 5 years of experience in embedded systems and firmware development
        Understanding of Computer Architecture, User Interfacing Technologies and Programming Languages
    """

    embeddings = create_embeddings()
    no_of_results = 3

    results = search_similar_documents(
        query, no_of_results, pinecone_index_name, embeddings
    )

    print(f"\nQuery: {query.strip()}")
    print(f"Number of results: {len(results)}\n")

    for i, (doc, score) in enumerate(results):
        print(f"Result {i + 1}:")
        print(f"  Score:   {score:.4f}")
        print(f"  Source:  {doc.metadata.get('source', 'Unknown')}")
        # Print first 200 characters of content
        print(f"  Content: {doc.page_content[:200]}...")
        print("=" * 50)
