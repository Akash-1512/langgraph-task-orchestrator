"""
tests/test_critique.py

Tests for the Critique Agent node.
Run with: python -m tests.test_critique
"""

from agents.critique import critique_node
from agents.state import AgentState

SAMPLE_ANALYTICS_RESULT = """
## Summary
Q1 OKR performance shows mixed results. Product adoption reached 72% on MAU target.
Revenue grew to $1.6M ARR (80% of target). Engineering reliability fell short with
99.7% uptime vs 99.9% target and 3 P1 incidents.

## Key Findings
- KR 1.1: 7,200 MAU vs 10,000 target (72% complete)
- KR 2.1: $1.6M ARR vs $2M target (80% complete)
- KR 3.1: 99.7% uptime vs 99.9% target (below threshold)

## Recommendations
- Reduce MAU target to 8,500 for Q2 and focus on activation
- Reallocate sales resources to enterprise deals
- Implement incident response playbook to eliminate P1s
"""


def test_critique_node():
    """Verify critique_node returns a valid CritiqueResult."""

    state: AgentState = {
        "query": "Analyze Q1 OKR performance and suggest adjustments for Q2",
        "messages": [],
        "plan": None,
        "research_context": [
            "KR 1.1: 7,200 MAU vs 10,000 target (72%)",
            "KR 2.1: $1.6M ARR vs $2M target (80%)",
            "KR 3.1: 99.7% uptime vs 99.9% target",
        ],
        "retrieved_sources": None,
        "analytics_result": SAMPLE_ANALYTICS_RESULT,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
    }

    result = critique_node(state)

    assert "critique" in result
    critique = result["critique"]
    assert "faithfulness_score" in critique
    assert "overall_score" in critique
    assert "passed_quality_gate" in critique
    assert 0.0 <= critique["overall_score"] <= 1.0

    print(f"✅ Critique scores:")
    print(f"   Faithfulness:    {critique['faithfulness_score']}")
    print(f"   Coherence:       {critique['coherence_score']}")
    print(f"   Task Completion: {critique['task_completion_score']}")
    print(f"   Overall:         {critique['overall_score']}")
    print(
        f"   Quality Gate:    {'✅ PASSED' if critique['passed_quality_gate'] else '❌ FAILED'}"
    )
    print(f"   Notes:           {critique['critique_notes']}")


if __name__ == "__main__":
    test_critique_node()
    print("\n✅ Critique agent tests passed.")
