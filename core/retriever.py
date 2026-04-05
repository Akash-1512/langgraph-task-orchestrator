"""
core/retriever.py — Provider-agnostic vector store abstraction.

FREE DEMO:        VECTOR_STORE=chroma  → Local ChromaDB in ./chroma_db/
AZURE PRODUCTION: VECTOR_STORE=qdrant  → Qdrant Cloud (1GB free) or
                                         Azure AI Search with vector index

Embedding model: BAAI/bge-small-en-v1.5 via FastEmbed (local, no API key)
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# FREE DEMO: FastEmbed runs locally — no API key, no cost
# AZURE PRODUCTION: Replace with Azure OpenAI text-embedding-3-small:
# from langchain_openai import AzureOpenAIEmbeddings
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment="text-embedding-3-small",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
# )
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_vector_store():
    """
    Returns configured vector store.

    FREE DEMO:        ChromaDB local — data persisted in ./chroma_db/
    AZURE PRODUCTION: Qdrant Cloud (QDRANT_URL + QDRANT_API_KEY)
                      OR Azure AI Search:
                      from langchain_community.vectorstores import AzureSearch
                      return AzureSearch(
                          azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
                          azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
                          index_name="sec-filings-index",
                          embedding_function=embeddings.embed_query,
                      )
    """
    store = os.getenv("VECTOR_STORE", "chroma").lower()

    if store == "chroma":
        # FREE DEMO: ChromaDB — local, persistent, zero setup
        from langchain_chroma import Chroma
        return Chroma(
            collection_name="sec_filings",
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )

    elif store == "qdrant":
        # PRODUCTION: Qdrant Cloud — 1GB free tier at cloud.qdrant.io
        # AZURE PRODUCTION: Deploy Qdrant on Azure Container Apps
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        return QdrantVectorStore(
            client=client,
            collection_name="sec_filings",
            embedding=embeddings,
        )

    else:
        raise ValueError(f"Unknown VECTOR_STORE: '{store}'. Valid: chroma | qdrant")


def get_retriever(k: int = 4, score_threshold: float = 0.3):
    """
    Returns configured retriever with similarity threshold.

    Args:
        k: Maximum number of chunks to retrieve
        score_threshold: Minimum similarity score (filters irrelevant chunks)

    AZURE PRODUCTION: Azure AI Search supports hybrid search (vector + keyword):
        return AzureSearch(...).as_retriever(
            search_type="hybrid",
            search_kwargs={"k": k}
        )
    """
    vs = get_vector_store()
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


def ingest_documents(documents: list[Document]) -> None:
    """
    Ingests documents into the configured vector store.

    FREE DEMO: ChromaDB — synchronous, local write
    AZURE PRODUCTION: Azure AI Search — use async batch upload:
        from azure.search.documents import SearchClient
        client = SearchClient(endpoint=..., index_name=..., credential=...)
        client.upload_documents(documents=batch)
    """
    vs = get_vector_store()
    vs.add_documents(documents)
    print(f"✅ Ingested {len(documents)} documents into {os.getenv('VECTOR_STORE', 'chroma')} vector store.")