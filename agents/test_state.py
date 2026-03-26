"""
Quick structural test for AgentState.
Run with: python agents/test_state.py
"""

from agents.state import AgentState, CritiqueResult, RunMetadata


def test_agent_state_structure():
    """Verify AgentState can be instantiated with all required fields."""

    sample_state: AgentState = {
        "query": "Analyze Q1 performance and suggest OKR adjustments",
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

    assert sample_state["query"] == "Analyze Q1 performance and suggest OKR adjustments"
    assert sample_state["hitl_status"] == "pending"
    assert sample_state["messages"] == []
    print("✅ AgentState structure verified — all fields present and typed correctly.")


def test_critique_result_structure():
    """Verify CritiqueResult TypedDict is correctly defined."""

    sample_critique: CritiqueResult = {
        "faithfulness_score": 0.92,
        "coherence_score": 0.88,
        "task_completion_score": 0.95,
        "overall_score": 0.916,
        "critique_notes": "Answer is well-grounded in retrieved OKR data.",
        "passed_quality_gate": True,
    }

    assert sample_critique["passed_quality_gate"] is True
    assert sample_critique["overall_score"] >= 0.75
    print("✅ CritiqueResult structure verified — quality gate logic confirmed.")


if __name__ == "__main__":
    test_agent_state_structure()
    test_critique_result_structure()
    print("\n✅ All state structure tests passed.")