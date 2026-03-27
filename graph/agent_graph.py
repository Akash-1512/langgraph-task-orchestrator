"""
graph/agent_graph.py

LangGraph StateGraph assembly for the langgraph-task-orchestrator.
Wires all agent nodes into a compiled, executable graph with:
  - Conditional edge after Critique (quality gate routing)
  - HITL interrupt checkpoint before final output
  - InMemorySaver checkpointer for state persistence across interrupts

Graph flow:
    START → planner → research → analytics → critique
                                    ↑              ↓ (quality gate failed)
                              hitl ← ←  ← ← ← ← ←
                                ↓ (approved)
                              END
                                ↓ (revised)
                           analytics (retry with feedback)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from agents.state import AgentState
from agents.planner import planner_node
from agents.research import research_node
from agents.analytics import analytics_node
from agents.critique import critique_node
from agents.hitl import hitl_node


def route_after_critique(state: AgentState) -> str:
    """
    Conditional edge function — runs after the Critique node.
    Routes to HITL if quality gate passed, back to analytics if failed.

    Returns:
        "hitl" if critique.passed_quality_gate is True
        "analytics" if critique.passed_quality_gate is False
    """
    critique = state.get("critique")
    if critique and critique.get("passed_quality_gate"):
        return "hitl"
    return "analytics"


def route_after_hitl(state: AgentState) -> str:
    """
    Conditional edge function — runs after the HITL node.
    Routes to END if approved, back to analytics if human requested revision.

    Returns:
        "end" if hitl_status == "approved"
        "analytics" if hitl_status == "revised"
    """
    status = state.get("hitl_status")
    if status == "approved":
        return "end"
    return "analytics"


def build_graph() -> StateGraph:
    """
    Builds and compiles the multi-agent OKR analytics StateGraph.

    Returns:
        Compiled LangGraph graph with MemorySaver checkpointer.
        The checkpointer is required for interrupt() to persist state.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("planner", planner_node)
    builder.add_node("research", research_node)
    builder.add_node("analytics", analytics_node)
    builder.add_node("critique", critique_node)
    builder.add_node("hitl", hitl_node)

    # ── Define edges ──────────────────────────────────────────────────────
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "research")
    builder.add_edge("research", "analytics")
    builder.add_edge("analytics", "critique")

    # Conditional edge: critique → hitl (passed) or analytics (failed)
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"hitl": "hitl", "analytics": "analytics"}
    )

    # Conditional edge: hitl → END (approved) or analytics (revised)
    builder.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {"end": END, "analytics": "analytics"}
    )

    # ── Compile with checkpointer (required for interrupt) ────────────────
    # checkpointer = MemorySaver()
    # return builder.compile(checkpointer=checkpointer)

    # SqliteSaver persists state to disk — survives server restarts.
    # MemorySaver is lost on restart — only suitable for testing.
    # DB_PATH can be overridden via env variable for Docker/cloud deployments.
    import sqlite3
    db_path = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)

def get_traced_graph():
    """
    Returns the compiled graph with Langfuse callbacks pre-attached.
    Use this for production — traces every node automatically.
    
    Usage:
        from graph.agent_graph import get_traced_graph
        from core.observability import get_callbacks
        traced = get_traced_graph()
        traced.stream(input, config={"callbacks": get_callbacks(), "configurable": {"thread_id": "..."}})
    """
    return graph


# Singleton graph instance — import this in api/ and ui/
graph = build_graph()