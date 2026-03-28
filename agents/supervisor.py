"""
agents/supervisor.py

Supervisor Agent — orchestrates the multi-agent pipeline by routing
state to the appropriate specialist agent based on current state.

In the current architecture, the Supervisor acts as a meta-router:
- Evaluates current state completeness
- Decides whether to retry analytics or proceed to critique
- Logs routing decisions for observability

This implements the Supervisor → Specialist pattern described in the
LangGraph multi-agent documentation.
"""

from agents.state import AgentState
from core.llm_client import get_llm
from langchain_core.messages import SystemMessage


SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor agent coordinating a multi-agent OKR analytics system.

Your job is to evaluate the current state and decide the next action.
Respond with ONLY one of these exact words:
- RESEARCH: if research_context is empty or insufficient
- ANALYTICS: if plan is ready but analytics_result is empty
- CRITIQUE: if analytics_result exists but critique is empty  
- HITL: if critique passed quality gate
- RETRY: if critique failed quality gate (retry analytics)
- END: if final_output is populated
"""


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor Agent node.
    Evaluates state and returns routing decision as a state update.

    Returns:
        dict with 'supervisor_decision' key for conditional edge routing
    """
    llm = get_llm()

    # Build state summary for supervisor
    state_summary = f"""Current state:
- query: {state.get('query', '')[:100]}
- plan: {'✅ present' if state.get('plan') else '❌ missing'}
- research_context: {'✅ present (' + str(len(state.get('research_context') or [])) + ' chunks)' if state.get('research_context') else '❌ missing'}
- analytics_result: {'✅ present' if state.get('analytics_result') else '❌ missing'}
- critique: {'✅ passed' if state.get('critique') and state['critique'].get('passed_quality_gate') else ('❌ failed' if state.get('critique') else '❌ missing')}
- hitl_status: {state.get('hitl_status', 'pending')}
- final_output: {'✅ present' if state.get('final_output') else '❌ missing'}
"""

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        SystemMessage(content=state_summary),
    ]

    response = llm.invoke(messages)
    decision = response.content.strip().upper()

    # Validate decision
    valid_decisions = {"RESEARCH", "ANALYTICS", "CRITIQUE", "HITL", "RETRY", "END"}
    if decision not in valid_decisions:
        decision = "ANALYTICS"  # Safe default

    print(f"🎯 Supervisor decision: {decision}")

    return {"supervisor_decision": decision}