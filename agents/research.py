"""
agents/research.py

Research Agent — second node in the LangGraph StateGraph.
Retrieves relevant document chunks from the vector store based on the plan.

Input state fields read:  plan, query
Output state fields written: research_context, retrieved_sources
"""

from langchain_core.messages import SystemMessage

from agents.state import AgentState
from core.retriever import get_retriever


def research_node(state: AgentState) -> dict:
    """
    Research Agent node function.
    Retrieves relevant context from vector store for each step in the plan.

    Args:
        state: Current AgentState — reads 'plan' and 'query' fields

    Returns:
        dict with updated 'research_context' and 'retrieved_sources' fields
    """
    retriever = get_retriever(k=4)

    plan = state.get("plan") or []
    query = state.get("query", "")

    # Build a combined search query from the original query + plan steps
    search_query = query + " " + " ".join(plan[:3])  # Use first 3 plan steps

    retrieved_docs = retriever.invoke(search_query)

    research_context = [doc.page_content for doc in retrieved_docs]
    retrieved_sources = [
        doc.metadata.get("source", "unknown") for doc in retrieved_docs
    ]

    return {
        "research_context": research_context,
        "retrieved_sources": retrieved_sources,
        "messages": [
            SystemMessage(
                content=f"Research Agent retrieved {len(research_context)} relevant chunks."
            )
        ],
    }
