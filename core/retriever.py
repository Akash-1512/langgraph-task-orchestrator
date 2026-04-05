"""
core/retriever.py

Provider-agnostic vector store retriever abstraction.
Returns a LangChain-compatible retriever based on VECTOR_STORE env variable.

Supported backends:
    chroma  → ChromaDB (local, in-process) — default for development
    qdrant  → Qdrant Cloud (free tier) — default for hosted demo

Usage:
    from core.retriever import get_retriever, ingest_documents
    retriever = get_retriever()

Swap backend by changing VECTOR_STORE in .env — zero code changes required.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

COLLECTION_NAME = "okr_knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model — runs fully locally


def get_embeddings():
    """Returns a LangChain-compatible FastEmbed embeddings instance."""
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_store():
    """
    Returns a LangChain-compatible vector store for the configured backend.
    Reads VECTOR_STORE from environment. Defaults to 'chroma' if not set.
    """
    backend = os.getenv("VECTOR_STORE", "chroma").lower()
    embeddings = get_embeddings()

    if backend == "chroma":
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )

    elif backend == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

    else:
        raise ValueError(
            f"Unsupported VECTOR_STORE: '{backend}'. "
            f"Supported values: chroma, qdrant"
        )


def get_retriever(k: int = 4, score_threshold: float = 0.3):
    """
    Returns a retriever that fetches top-k relevant chunks.

    Args:
        k: Number of document chunks to retrieve. Default 4.

    Returns:
        LangChain retriever instance
    """
    vs = get_vector_store()
    # Use similarity_score_threshold instead of plain similarity search
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold,
        }
    )

def ingest_documents(documents: list[Document]) -> None:
    """
    Ingests a list of LangChain Document objects into the vector store.
    Used for loading OKR knowledge base documents.

    Args:
        documents: List of LangChain Document objects with page_content and metadata
    """
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    print(f"✅ Ingested {len(documents)} documents into {os.getenv('VECTOR_STORE', 'chroma')} vector store.")
    