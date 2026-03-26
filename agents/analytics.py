"""
agents/analytics.py

Analytics Agent — third node in the LangGraph StateGraph.
Produces a structured analytical response grounded in retrieved OKR context.

Input state fields read:  query, plan, research_context, retrieved_sources, hitl_feedback
Output state fields written: analytics_result, messages
"""

from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState
from core.llm_client import get_llm


ANALYTICS_SYSTEM_PROMPT = """You are a senior OKR analytics agent. Your job is to analyze 
business performance data and produce a structured, actionable report.

Rules:
- Ground every claim in the provided context. Never hallucinate data.
- Structure your response with clear sections: Summary, Key Findings, Recommendations.
- Be specific — cite exact numbers and percentages from the context.
- If revision feedback is provided, address it directly in your response.
- Keep the response concise but complete — aim for 300-400 words.
"""


def analytics_node(state: AgentState) -> dict:
    """
    Analytics Agent node function.
    Produces a structured analytical response grounded in retrieved context.

    Args:
        state: Current AgentState — reads query, plan, research_context,
               retrieved_sources, and hitl_feedback fields

    Returns:
        dict with updated 'analytics_result' and 'messages' fields
    """
    llm = get_llm()

    plan = state.get("plan") or []
    research_context = state.get("research_context") or []
    hitl_feedback = state.get("hitl_feedback")

    # Format retrieved context for the prompt
    context_text = "\n\n".join([
        f"[Source: {state.get('retrieved_sources', ['unknown'])[i] if state.get('retrieved_sources') else 'unknown'}]\n{chunk}"
        for i, chunk in enumerate(research_context)
    ])

    # Build the user message — include HITL feedback if this is a revision
    user_content = f"""Query: {state['query']}

Plan:
{chr(10).join(f"- {step}" for step in plan)}

Retrieved Context:
{context_text}
"""

    if hitl_feedback:
        user_content += f"\nRevision Feedback from Human Reviewer:\n{hitl_feedback}\n"
        user_content += "\nPlease revise your analysis addressing the feedback above."

    messages = [
        SystemMessage(content=ANALYTICS_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]

    from core.observability import get_callbacks
    callbacks = get_callbacks()
    response = llm.invoke(messages, config={"callbacks": callbacks})

    return {
        "analytics_result": response.content.strip(),
        "messages": [response]
    }