"""
agents/critique.py

Critique Agent — fourth node in the LangGraph StateGraph.
Acts as LLM-as-Judge, scoring the analytics output before it reaches
the human reviewer.

Industry-standard fix applied:
- Uses with_structured_output() + Pydantic model instead of regex parsing.
  LLM returns a typed CritiqueResult object — no regex, no case sensitivity,
  no silent 0.0 defaults from malformed output.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.state import AgentState, CritiqueResult
from core.llm_client import get_llm
from core.observability import get_callbacks

# ── Pydantic schema for structured LLM output ─────────────────────────────────


class CritiqueScore(BaseModel):
    """Structured critique scores returned directly by the LLM."""

    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="Are all claims in the response grounded in the retrieved context?",
    )
    coherence: float = Field(
        ge=0.0,
        le=1.0,
        description="Is the response logically structured and internally consistent?",
    )
    task_completion: float = Field(
        ge=0.0,
        le=1.0,
        description="Does the response fully address the original query?",
    )
    notes: str = Field(
        description="One sentence explaining the most significant quality issue, if any."
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────

CRITIQUE_SYSTEM_PROMPT = """You are an impartial AI quality reviewer.

Evaluate the analytics report against the retrieved context and original query.
Score each dimension from 0.0 (completely fails) to 1.0 (perfect).

Scoring guide:
- faithfulness:      Are ALL claims directly supported by the retrieved context?
                     Penalise any claim not traceable to the provided sources.
- coherence:         Is the report logically structured, non-repetitive, and clear?
- task_completion:   Does the report fully answer what the query asked for?

Be strict. A score of 0.9+ means the report is publication-ready."""


# ── Node ───────────────────────────────────────────────────────────────────────


def critique_node(state: AgentState) -> dict:
    """
    Critique Agent node — LLM-as-Judge using structured output.

    Uses with_structured_output(CritiqueScore) so the LLM returns
    a typed Pydantic object. Eliminates all regex parsing, case-sensitivity
    issues, and silent 0.0 defaults.

    Args:
        state: Current AgentState — reads query, research_context,
               analytics_result fields.

    Returns:
        dict with updated 'critique' field (CritiqueResult TypedDict).
    """
    llm = get_llm()

    # Bind structured output schema — LLM must return valid CritiqueScore JSON
    structured_llm = llm.with_structured_output(CritiqueScore)

    query = state.get("query", "")
    research_context = state.get("research_context") or []
    analytics_result = state.get("analytics_result", "")

    context_text = "\n\n".join(
        [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(research_context)]
    )

    user_content = f"""Original Query:
{query}

Retrieved Context (source of truth):
{context_text}

Analytics Report to evaluate:
{analytics_result}

Score this report on faithfulness, coherence, and task_completion.
Provide one sentence of notes on the most significant quality issue."""

    messages = [
        SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    callbacks = get_callbacks()
    scored: CritiqueScore = structured_llm.invoke(
        messages, config={"callbacks": callbacks}
    )

    # Weighted overall score
    overall = round(
        scored.faithfulness * 0.4
        + scored.coherence * 0.3
        + scored.task_completion * 0.3,
        4,
    )
    passed = overall >= 0.75

    critique: CritiqueResult = {
        "faithfulness_score": scored.faithfulness,
        "coherence_score": scored.coherence,
        "task_completion_score": scored.task_completion,
        "overall_score": overall,
        "passed_quality_gate": passed,
        "critique_notes": scored.notes,
    }

    print(f"✅ Critique scores:")
    print(f"   Faithfulness:    {scored.faithfulness}")
    print(f"   Coherence:       {scored.coherence}")
    print(f"   Task Completion: {scored.task_completion}")
    print(f"   Overall:         {overall}")
    print(f"   Quality Gate:    {'✅ PASSED' if passed else '❌ FAILED'}")
    if scored.notes:
        print(f"   Notes:           {scored.notes}")

    return {"critique": critique}
