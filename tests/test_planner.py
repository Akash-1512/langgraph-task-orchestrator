"""
tests/test_planner.py

Tests for the Planner Agent node.
Run with: python -m tests.test_planner
"""

from agents.state import AgentState
from agents.planner import planner_node


def test_planner_node():
    """Verify planner_node returns a valid plan from a real LLM call."""

    initial_state: AgentState = {
        "query": "Analyze Q1 OKR performance and suggest adjustments for Q2",
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

    result = planner_node(initial_state)

    assert "plan" in result, "planner_node must return 'plan' key"
    assert isinstance(result["plan"], list), "plan must be a list"
    assert len(result["plan"]) > 0, "plan must have at least one step"
    assert "messages" in result, "planner_node must return 'messages' key"

    print(f"✅ Planner returned {len(result['plan'])} sub-tasks:")
    for i, step in enumerate(result["plan"], 1):
        print(f"   {i}. {step}")


if __name__ == "__main__":
    test_planner_node()
    print("\n✅ Planner agent tests passed.")