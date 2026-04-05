"""
agents/state.py

Defines the AgentState TypedDict — the single source of truth for all
state flowing through the LangGraph StateGraph. Every node reads from
and writes to this object. No node communicates with another directly.

Design principle: explicit over implicit. Every field has a type annotation
and a comment explaining its contract. This makes the graph traceable,
debuggable, and self-documenting.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class RunMetadata(TypedDict):
    """Tracks execution metadata for observability and debugging."""
    run_id: str
    started_at: str
    completed_at: Optional[str]
    llm_provider: str
    model_name: str
    total_tokens: int


class CritiqueResult(TypedDict):
    """Output from the LLM-as-Judge Critique agent."""
    faithfulness_score: float      # 0.0 - 1.0: Is the answer grounded in retrieved context?
    coherence_score: float         # 0.0 - 1.0: Is the answer logically consistent?
    task_completion_score: float   # 0.0 - 1.0: Does the answer fully address the query?
    overall_score: float           # Weighted average of above scores
    critique_notes: str            # Natural language explanation of scores
    passed_quality_gate: bool      # True if overall_score >= 0.75


class AgentState(TypedDict):
    """
    Central state object for the langgraph-task-orchestrator multi-agent system.

    State flows through nodes in this order:
    Planner → Research → Analytics → Critique → HITL Checkpoint → Output

    LangGraph uses this TypedDict to:
    1. Type-check state transitions at graph compile time
    2. Persist state to checkpointer before HITL interrupt
    3. Resume from exact state after human approval

    Annotated[list, add_messages] tells LangGraph to APPEND to messages
    rather than overwrite — this preserves the full conversation history.
    """

    # ── Input ─────────────────────────────────────────────────────────────
    query: str
    # The original business query from the user.
    # Example: "Analyze Q1 performance and suggest OKR adjustments"

    messages: Annotated[list, add_messages]
    # Full message history. LangGraph appends to this automatically.
    # Never overwrite this field directly.

    # ── Planner Agent Output ──────────────────────────────────────────────
    plan: Optional[list[str]]
    # Ordered list of sub-tasks decomposed by the Planner agent.
    # Example: ["Retrieve Q1 OKR data", "Calculate KR completion rates",
    #           "Identify underperforming objectives", "Suggest adjustments"]

    # ── Research Agent Output ─────────────────────────────────────────────
    research_context: Optional[list[str]]
    # List of retrieved document chunks from the vector store.
    # Used by Analytics agent and evaluated by RAGAS faithfulness metric.

    retrieved_sources: Optional[list[str]]
    # Source document names/IDs for the retrieved chunks.
    # Used for citation and RAGAS context recall evaluation.

    # ── Analytics Agent Output ────────────────────────────────────────────
    analytics_result: Optional[str]
    # Natural language analytics output from the Analytics agent.
    # Grounded in research_context — cites specific retrieved data points.

    # ── Critique Agent Output ─────────────────────────────────────────────
    critique: Optional[CritiqueResult]
    # LLM-as-Judge evaluation of the analytics_result.
    # If critique.passed_quality_gate is False, graph routes back to Analytics.

    # ── HITL Checkpoint ───────────────────────────────────────────────────
    hitl_status: Optional[str]
    # Values: "pending" | "approved" | "revised"
    # "pending"  → graph is paused at HITL interrupt, awaiting human input
    # "approved" → human approved, graph proceeds to final output
    # "revised"  → human provided feedback, graph routes back to Analytics

    hitl_feedback: Optional[str]
    # Human reviewer's revision notes when hitl_status == "revised".
    # Passed back to Analytics agent as additional context for revision.

    # ── Final Output ──────────────────────────────────────────────────────
    final_output: Optional[str]
    # The approved, finalized response delivered to the user.
    # Only populated after hitl_status == "approved".

    # ── Observability ─────────────────────────────────────────────────────
    run_metadata: Optional[RunMetadata]
    # Execution metadata for Langfuse tracing and MLflow logging.

    error: Optional[str]
    # Error message if any node fails. Enables graceful failure handling
    # without crashing the entire graph.

    retry_count: Optional[int]   # Guards against infinite quality-gate loops
    