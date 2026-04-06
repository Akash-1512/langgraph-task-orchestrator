"""
tests/test_analytics.py

Unit test for Analytics Agent using FakeListChatModel.
Zero real LLM calls. Zero tokens consumed.

Real LLM test: tests/integration/test_analytics_integration.py
"""

from langchain_community.chat_models.fake import FakeListChatModel

import agents.analytics as analytics_module
from agents.analytics import analytics_node
from agents.state import AgentState

FAKE_ANALYTICS_RESPONSE = """## Summary
Q1 OKR performance shows mixed results across all three objectives.
MAU achieved 72% of target. ARR reached 80% of target at $1.6M.

## Key Findings
- Key Result 1.1: 7,200 MAU vs 10,000 target (72% complete)
- Key Result 2.1: $1.6M ARR vs $2M target (80% complete)
- Key Result 3.1: 99.7% uptime vs 99.9% target (below threshold)
- Key Result 3.2: 3 P1 incidents vs zero target (critical miss)

## Recommendations
- Reduce MAU target to 8,500 for Q2 given current growth trajectory
- Split enterprise deal target into monthly milestones of 15 deals
- Allocate dedicated SRE sprint to uptime and P1 incident reduction"""


def _make_state() -> AgentState:
    return {
        "query": "Analyze Q1 OKR performance and suggest Q2 adjustments",
        "messages": [],
        "plan": [
            "Retrieve Q1 OKR performance data",
            "Calculate key result completion rates",
            "Identify underperforming objectives",
            "Analyze root causes",
            "Suggest Q2 adjustments",
        ],
        "research_context": [
            "Key Result 1.1: Achieve 10,000 MAU — Result: 7,200 (72% complete).",
            "Key Result 2.1: Achieve $2M ARR — Result: $1.6M (80% complete).",
            "Key Result 3.1: Achieve 99.9% uptime — Result: 99.7% (below target).",
            "Key Result 3.2: Reduce P1 incidents to zero — Result: 3 incidents.",
        ],
        "retrieved_sources": [
            "q1_okr_report.pdf",
            "q1_sales_okr.pdf",
            "q1_engineering_okr.pdf",
            "okr_best_practices.pdf",
        ],
        "analytics_result": None,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
        "retry_count": 0,
    }


def test_analytics_node_unit():
    """
    Unit test: Analytics produces structured output from mocked LLM.
    Verifies Summary/Key Findings/Recommendations structure.
    Zero real LLM calls.
    """
    fake_llm = FakeListChatModel(responses=[FAKE_ANALYTICS_RESPONSE])
    original_get_llm = analytics_module.get_llm
    analytics_module.get_llm = lambda **kwargs: fake_llm

    try:
        result = analytics_node(_make_state())

        assert "analytics_result" in result
        assert len(result["analytics_result"]) > 100
        assert "Summary" in result["analytics_result"]
        assert "Key Findings" in result["analytics_result"]
        assert "Recommendations" in result["analytics_result"]
        assert result["retry_count"] == 1

        print(f"✅ Analytics unit test passed")
        print(f"   Output length: {len(result['analytics_result'])} chars")
        print(f"   Retry count incremented to: {result['retry_count']}")

    finally:
        analytics_module.get_llm = original_get_llm


def test_analytics_with_hitl_feedback():
    """Unit test: Analytics incorporates HITL revision feedback."""
    fake_llm = FakeListChatModel(responses=[FAKE_ANALYTICS_RESPONSE])
    original_get_llm = analytics_module.get_llm
    analytics_module.get_llm = lambda **kwargs: fake_llm

    try:
        state = _make_state()
        state["hitl_feedback"] = "Focus more on Q2 revenue projections"
        result = analytics_node(state)

        assert "analytics_result" in result
        print("✅ Analytics HITL feedback test passed")

    finally:
        analytics_module.get_llm = original_get_llm


if __name__ == "__main__":
    print("🔍 Running Analytics unit tests (mocked LLM)...")
    test_analytics_node_unit()
    test_analytics_with_hitl_feedback()
    print("\n✅ All Analytics unit tests passed — zero tokens consumed.")
