"""
tools/retrieval_tool.py

LangChain tool definitions for the Research Agent.
Wraps the vector store retriever as a callable LangChain tool
so agents can invoke it via tool calling.
"""

from langchain_core.tools import tool
from core.retriever import get_retriever


@tool
def search_sec_filings(query: str) -> str:
    """
    Search SEC EDGAR filings for relevant financial information.
    
    Use this tool to retrieve information from real SEC 10-K and 10-Q filings
    from Apple, Microsoft, Google, Salesforce, Netflix, Tesla, Amazon, and Meta.
    
    Args:
        query: Natural language search query about company financials, 
               KPIs, risks, strategy, or performance metrics.
    
    Returns:
        Relevant passages from SEC filings as a formatted string.
    """
    retriever = get_retriever(k=4)
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant SEC filing information found."
    
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        company = doc.metadata.get("company", "unknown")
        filing_type = doc.metadata.get("filing_type", "unknown")
        results.append(
            f"[{i}] Source: {company} {filing_type} ({source})\n{doc.page_content}"
        )
    
    return "\n\n".join(results)


@tool  
def search_okr_best_practices(query: str) -> str:
    """
    Search for OKR best practices and framework guidance.
    
    Use this tool when you need guidance on OKR methodology,
    goal-setting frameworks, or performance management best practices.
    
    Args:
        query: Question about OKR methodology or best practices.
    
    Returns:
        Relevant OKR guidance from the knowledge base.
    """
    retriever = get_retriever(k=2)
    docs = retriever.invoke(f"OKR best practices {query}")
    
    if not docs:
        return "No OKR best practice information found."
    
    return "\n\n".join([doc.page_content for doc in docs])


# Tool registry — import this in agent nodes
SEC_TOOLS = [search_sec_filings, search_okr_best_practices]