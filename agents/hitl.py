"""
agents/hitl.py

HITL (Human-in-the-Loop) Checkpoint node for the LangGraph StateGraph.
Uses LangGraph's interrupt() to pause execution and await human input.

Input state fields read:  analytics_result, critique
Output state fields written: hitl_status, hitl_feedback, final_output
"""

from langgraph.types import interrupt
from agents.state import AgentState


def hitl_node(state: AgentState) -> dict:
    """
    HITL Checkpoint node.
    Pauses the graph using interrupt() and waits for human approval.

    The graph state is persisted to the checkpointer before pausing.
    After human input is provided via Command(resume=...), the graph
    resumes from exactly this point with zero state loss.

    Args:
        state: Current AgentState — reads analytics_result and critique

    Returns:
        dict with updated hitl_status, hitl_feedback, and final_output
    """
    critique = state.get("critique")
    analytics_result = state.get("analytics_result", "")

    # Present the result and scores to the human reviewer
    interrupt_payload = {
        "analytics_result": analytics_result,
        "scores": {
            "faithfulness": critique["faithfulness_score"] if critique else None,
            "coherence": critique["coherence_score"] if critique else None,
            "task_completion": critique["task_completion_score"] if critique else None,
            "overall": critique["overall_score"] if critique else None,
        },
        "critique_notes": critique["critique_notes"] if critique else "",
        "instructions": "Reply with 'approve' to finalize, or provide revision feedback."
    }

    # Pause graph execution — waits for Command(resume=<human_input>)
    human_input = interrupt(interrupt_payload)

    # Process human response
    if isinstance(human_input, str) and human_input.strip().lower() == "approve":
        return {
            "hitl_status": "approved",
            "hitl_feedback": None,
            "final_output": analytics_result,
        }
    else:
        return {
            "hitl_status": "revised",
            "hitl_feedback": str(human_input),
            "final_output": None,
        }