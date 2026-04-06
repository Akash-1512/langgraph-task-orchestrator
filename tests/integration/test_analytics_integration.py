"""
tests/integration/test_analytics_integration.py

Integration test for Analytics Agent using real Groq LLM.
Run manually only — NOT on every push.

Run with:
    python -m tests.integration.test_analytics_integration
"""

from agents.analytics import analytics_node
from agents.state import AgentState


def test_analytics_real_llm():
    state: AgentState = {
        "query": "Analyze Q1 OKR performance and suggest Q2 adjustments",
        "messages": [],
        "plan": ["Retrieve data", "Analyze KRs", "Recommend adjustments"],
        "research_context": [
            "Key Result 1.1: Achieve 10,000 MAU — Result: 7,200 (72%).",
            "Key Result 2.1: Achieve $2M ARR — Result: $1.6M (80%).",
        ],
        "retrieved_sources": ["q1_okr_report.pdf"],
        "analytics_result": None,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
        "retry_count": 0,
    }

    result = analytics_node(state)
    assert "analytics_result" in result
    assert len(result["analytics_result"]) > 200
    print(f"✅ Real LLM analytics: {len(result['analytics_result'])} chars")
    print(f"   Preview: {result['analytics_result'][:200]}...")


if __name__ == "__main__":
    print("🔍 Running Analytics INTEGRATION test (real Groq LLM)...")
    test_analytics_real_llm()
    print("\n✅ Integration test passed.")
