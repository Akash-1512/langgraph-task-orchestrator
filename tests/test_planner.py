"""
tests/test_planner.py

Unit test for Planner Agent using FakeListChatModel.

Industry-standard: unit tests NEVER make real LLM calls.
- Zero tokens consumed
- Deterministic output
- Runs in CI without API keys

Real LLM integration test: tests/integration/test_planner_integration.py
"""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages import AIMessage

import agents.planner as planner_module
from agents.planner import planner_node
from agents.state import AgentState

FAKE_PLAN_RESPONSE = """1. Retrieve Q1 OKR performance data from the knowledge base
2. Calculate key result completion rates for each objective
3. Identify underperforming objectives below 70 percent completion
4. Analyze root causes for underperformance using retrieved context
5. Suggest specific measurable OKR adjustments for Q2"""


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


def test_planner_node_unit(monkeypatch=None):
    """
    Unit test: Planner parses LLM output into a valid plan list.
    Uses FakeListChatModel — zero real LLM calls, zero tokens.
    """
    fake_llm = FakeListChatModel(responses=[FAKE_PLAN_RESPONSE])

    # Patch get_llm to return the fake model
    original_get_llm = planner_module.get_llm
    planner_module.get_llm = lambda **kwargs: fake_llm

    try:
        state = _make_state("Analyze Q1 OKR performance and suggest Q2 adjustments")
        result = planner_node(state)

        assert "plan" in result, "Expected 'plan' key in result"
        assert isinstance(result["plan"], list), "Plan must be a list"
        assert (
            len(result["plan"]) == 5
        ), f"Expected 5 sub-tasks, got {len(result['plan'])}"
        assert all(isinstance(step, str) for step in result["plan"])
        assert all(len(step) > 5 for step in result["plan"])

        print(f"✅ Planner unit test passed — {len(result['plan'])} sub-tasks parsed")
        for i, step in enumerate(result["plan"], 1):
            print(f"   {i}. {step}")

    finally:
        planner_module.get_llm = original_get_llm


def test_planner_empty_query():
    """Unit test: Planner handles minimal query without crashing."""
    fake_llm = FakeListChatModel(
        responses=["1. Research the topic\n2. Analyze findings"]
    )
    original_get_llm = planner_module.get_llm
    planner_module.get_llm = lambda **kwargs: fake_llm

    try:
        state = _make_state("OKR review")
        result = planner_node(state)
        assert "plan" in result
        assert len(result["plan"]) >= 1
        print("✅ Planner empty query test passed")
    finally:
        planner_module.get_llm = original_get_llm


if __name__ == "__main__":
    print("🔍 Running Planner unit tests (mocked LLM)...")
    test_planner_node_unit()
    test_planner_empty_query()
    print("\n✅ All Planner unit tests passed — zero tokens consumed.")
