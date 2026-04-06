"""
agents/planner.py

Planner Agent — first node in the LangGraph StateGraph.
Decomposes a high-level business query into an ordered list of sub-tasks.

Input state fields read:  query
Output state fields written: plan, messages
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from core.llm_client import get_llm

PLANNER_SYSTEM_PROMPT = """You are a strategic planning agent for a multi-agent OKR analytics system.

Your job is to decompose a high-level business query into a clear, ordered list of sub-tasks 
that specialist agents will execute sequentially.

Rules:
- Output ONLY a numbered list of sub-tasks. No preamble, no explanation.
- Each sub-task must be a single, actionable instruction.
- Maximum 5 sub-tasks. Be concise.
- Sub-tasks must logically build on each other.

Example output for "Analyze Q1 performance and suggest OKR adjustments":
1. Retrieve Q1 OKR performance data from the knowledge base
2. Calculate key result completion rates for each objective
3. Identify underperforming objectives (below 70% completion)
4. Analyze root causes for underperformance using retrieved context
5. Suggest specific, measurable OKR adjustments for Q2
"""


def planner_node(state: AgentState) -> dict:
    """
    Planner Agent node function.
    Called by LangGraph when the graph enters the 'planner' node.

    Args:
        state: Current AgentState — reads 'query' field

    Returns:
        dict with updated 'plan' and 'messages' fields
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Business query: {state['query']}"),
    ]

    from core.observability import get_callbacks

    callbacks = get_callbacks()
    response = llm.invoke(messages, config={"callbacks": callbacks})

    # Parse the numbered list response into a clean Python list
    plan_text = response.content.strip()
    plan_lines = [
        line.strip().lstrip("0123456789").lstrip(". ").strip()
        for line in plan_text.split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]
    return {
        "plan": plan_lines,
        "messages": [HumanMessage(content=state["query"]), response],
    }
