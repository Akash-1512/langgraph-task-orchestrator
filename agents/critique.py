"""
agents/critique.py

Critique Agent — LLM-as-Judge node in the LangGraph StateGraph.
Evaluates the analytics_result on faithfulness, coherence, and task completion.

Input state fields read:  query, analytics_result, research_context
Output state fields written: critique, messages
"""

from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState, CritiqueResult
from core.llm_client import get_llm


CRITIQUE_SYSTEM_PROMPT = """You are an impartial AI judge evaluating the quality of an analytics report.

Score the report on three dimensions (0.0 to 1.0 each):
1. faithfulness: Are all claims grounded in the provided context? No hallucinations?
2. coherence: Is the response logically structured and internally consistent?
3. task_completion: Does the response fully address the original query?

Respond ONLY in this exact format — no preamble, no explanation outside the format:
FAITHFULNESS: <score>
COHERENCE: <score>
TASK_COMPLETION: <score>
NOTES: <one sentence explanation>
"""


def _parse_critique_response(response_text: str, threshold: float = 0.75) -> CritiqueResult:
    """Parses the LLM judge response into a CritiqueResult TypedDict."""
    lines = response_text.strip().split("\n")
    scores = {}
    notes = ""

    for line in lines:
        if line.startswith("FAITHFULNESS:"):
            scores["faithfulness_score"] = float(line.split(":")[1].strip())
        elif line.startswith("COHERENCE:"):
            scores["coherence_score"] = float(line.split(":")[1].strip())
        elif line.startswith("TASK_COMPLETION:"):
            scores["task_completion_score"] = float(line.split(":")[1].strip())
        elif line.startswith("NOTES:"):
            notes = line.split(":", 1)[1].strip()

    overall = round(
        (scores.get("faithfulness_score", 0) * 0.4 +
         scores.get("coherence_score", 0) * 0.3 +
         scores.get("task_completion_score", 0) * 0.3),
        3
    )

    return CritiqueResult(
        faithfulness_score=scores.get("faithfulness_score", 0.0),
        coherence_score=scores.get("coherence_score", 0.0),
        task_completion_score=scores.get("task_completion_score", 0.0),
        overall_score=overall,
        critique_notes=notes,
        passed_quality_gate=overall >= threshold,
    )


def critique_node(state: AgentState) -> dict:
    """
    Critique Agent node function — LLM-as-Judge.
    Evaluates analytics_result quality and sets passed_quality_gate.

    Args:
        state: Current AgentState — reads query, analytics_result, research_context

    Returns:
        dict with updated 'critique' and 'messages' fields
    """
    llm = get_llm()

    context_text = "\n\n".join(state.get("research_context") or [])

    user_content = f"""Original Query: {state['query']}

Retrieved Context:
{context_text}

Analytics Report to Evaluate:
{state.get('analytics_result', '')}
"""

    messages = [
        SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]

    response = llm.invoke(messages)
    critique = _parse_critique_response(response.content)

    return {
        "critique": critique,
        "messages": [response]
    }