"""
tests/integration/test_planner_integration.py

Integration test for Planner Agent using real Groq LLM.
Only run manually or on scheduled CI — NOT on every push.

Run with:
    python -m tests.integration.test_planner_integration

Requires: GROQ_API_KEY in .env
"""

from agents.planner import planner_node
from agents.state import AgentState


def _make_state(query: str) -> AgentState:
    return {
        "query": query,
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
        "retry_count": 0,
    }


def test_planner_real_llm():
    """Integration test: real Groq LLM call — consumes tokens."""
    state = _make_state("Analyze Apple Q1 revenue performance from SEC filings")
    result = planner_node(state)

    assert "plan" in result
    assert len(result["plan"]) >= 3
    print(f"✅ Real LLM planner returned {len(result['plan'])} sub-tasks")
    for i, step in enumerate(result["plan"], 1):
        print(f"   {i}. {step}")


if __name__ == "__main__":
    print("🔍 Running Planner INTEGRATION test (real Groq LLM)...")
    test_planner_real_llm()
    print("\n✅ Integration test passed.")
