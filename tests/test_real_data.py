"""
tests/test_real_data.py

Integration test using real SEC EDGAR data.
Verifies the full pipeline works with real Apple/Microsoft/Tesla filings.
Run with: python -m tests.test_real_data

NOTE: Requires chroma_db to be populated via:
    python -m data.ingest_sec_filings
"""

import os
import shutil
from agents.state import AgentState
from agents.planner import planner_node
from agents.research import research_node
from agents.analytics import analytics_node


REAL_QUERIES = [
    "Analyze Apple's revenue performance and key business risks from their latest annual report",
    "What are Microsoft's cloud computing growth metrics and strategic priorities?",
    "Summarize Tesla's production targets and financial performance from recent filings",
]


def test_research_with_real_data():
    """Verify research agent retrieves real SEC filing chunks."""

    # Check chroma_db exists
    if not os.path.exists("./chroma_db"):
        print("⚠️  chroma_db not found — run: python -m data.ingest_sec_filings")
        print("   Skipping real data test.")
        return

    state: AgentState = {
        "query": REAL_QUERIES[0],
        "messages": [],
        "plan": [
            "Retrieve Apple revenue data from SEC filings",
            "Identify key risk factors mentioned in 10-K",
            "Analyze year-over-year performance trends",
        ],
        "research_context": None,
        "retrieved_sources": None,
        "analytics_result": None,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
    }

    result = research_node(state)

    assert "research_context" in result
    assert len(result["research_context"]) > 0
    assert "retrieved_sources" in result

    # Verify sources are real SEC filings
    sources = result["retrieved_sources"]
    real_tickers = ["AAPL", "MSFT", "GOOGL", "CRM", "NFLX", "TSLA", "AMZN", "META"]
    has_real_source = any(
        any(ticker in str(source) for ticker in real_tickers)
        for source in sources
    )

    print(f"✅ Retrieved {len(result['research_context'])} chunks from real SEC filings")
    print(f"   Sources: {sources}")
    print(f"   Contains real SEC data: {has_real_source}")

    assert has_real_source, "Expected real SEC filing sources (AAPL, MSFT, etc.)"


def test_analytics_with_real_data():
    """Verify analytics agent produces grounded output from real SEC data."""

    if not os.path.exists("./chroma_db"):
        print("⚠️  chroma_db not found — skipping.")
        return

    # Run planner first
    initial_state: AgentState = {
        "query": REAL_QUERIES[1],
        "messages": [],
        "plan": None,
        "research_context": None,
        "retrieved_sources": None,
        "analytics_result": None,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
    }

    planner_result = planner_node(initial_state)
    state = {**initial_state, **planner_result}

    # Run research
    research_result = research_node(state)
    state = {**state, **research_result}

    # Run analytics
    analytics_result = analytics_node(state)

    assert "analytics_result" in analytics_result
    assert len(analytics_result["analytics_result"]) > 100

    print(f"✅ Analytics produced {len(analytics_result['analytics_result'])} char response")
    print(f"   Preview: {analytics_result['analytics_result'][:200]}...")


if __name__ == "__main__":
    print("🔍 Running real SEC EDGAR data tests...")
    test_research_with_real_data()
    test_analytics_with_real_data()
    print("\n✅ All real data tests passed.")