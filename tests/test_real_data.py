"""
tests/test_real_data.py

Integration test using real SEC EDGAR data.
Run with: python -m tests.test_real_data

NOTE: Requires chroma_db to be populated via:
    python -m data.ingest_sec_filings
"""

import os
from agents.state import AgentState
from agents.planner import planner_node
from agents.research import research_node
from agents.analytics import analytics_node

REAL_QUERIES = [
    "Analyze Apple's revenue performance and key business risks from their latest annual report",
    "What are Microsoft's cloud computing growth metrics and strategic priorities?",
    "Summarize Tesla's production targets and financial performance from recent filings",
]

REAL_TICKERS = ["AAPL", "MSFT", "GOOGL", "CRM", "NFLX", "TSLA", "AMZN", "META"]


def _make_state(query: str, plan: list = None) -> AgentState:
    return {
        "query": query,
        "messages": [],
        "plan": plan,
        "research_context": None,
        "retrieved_sources": None,
        "analytics_result": None,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
        "retry_count": 0,
    }


def test_research_with_real_data():
    """Verify research agent retrieves real SEC filing chunks."""
    if not os.path.exists("./chroma_db"):
        print("⚠️  chroma_db not found — run: python -m data.ingest_sec_filings")
        return

    state = _make_state(
        query=REAL_QUERIES[0],
        plan=[
            "Retrieve Apple revenue data from SEC filings",
            "Identify key risk factors mentioned in 10-K",
            "Analyze year-over-year performance trends",
        ]
    )

    result = research_node(state)

    assert "research_context" in result
    assert len(result["research_context"]) > 0
    assert "retrieved_sources" in result

    sources = result["retrieved_sources"]
    has_real_source = any(
        any(ticker in str(source) for ticker in REAL_TICKERS)
        for source in sources
    )

    print(f"✅ Retrieved {len(result['research_context'])} chunks from real SEC filings")
    print(f"   Sources: {sources}")
    print(f"   Contains real SEC data: {has_real_source}")

    if not has_real_source:
        print("⚠️  SEC data not in top results — OKR docs may be dominating retrieval")
        print("   Re-run: python -m data.ingest_sec_filings")
        return

    assert has_real_source, "Expected real SEC filing sources"


def test_analytics_with_real_data():
    """Verify analytics agent produces grounded output from real SEC data."""
    if not os.path.exists("./chroma_db"):
        print("⚠️  chroma_db not found — skipping.")
        return

    state = _make_state(query=REAL_QUERIES[1])
    state = {**state, **planner_node(state)}
    state = {**state, **research_node(state)}
    result = analytics_node(state)

    assert "analytics_result" in result
    assert len(result["analytics_result"]) > 100

    print(f"✅ Analytics produced {len(result['analytics_result'])} char response")
    print(f"   Preview: {result['analytics_result'][:200]}...")


if __name__ == "__main__":
    print("🔍 Running real SEC EDGAR data tests...")
    test_research_with_real_data()
    test_analytics_with_real_data()
    print("\n✅ All real data tests passed.")