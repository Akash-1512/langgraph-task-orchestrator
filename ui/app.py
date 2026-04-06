"""
ui/app.py — Streamlit frontend for langgraph-task-orchestrator
Run with: streamlit run ui/app.py
"""

import time
import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="OKR Analytics Agent", layout="wide", page_icon="🤖")

st.title("🤖 Multi-Agent OKR Analytics System")
st.caption("Powered by LangGraph · Groq · Real SEC EDGAR Data · LLM-as-Judge")

# Session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "hitl_data" not in st.session_state:
    st.session_state.hitl_data = None
if "nodes_completed" not in st.session_state:
    st.session_state.nodes_completed = []
if "final_output" not in st.session_state:
    st.session_state.final_output = None

# Agent graph visualization
GRAPH_NODES = ["planner", "research", "analytics", "critique", "hitl"]


def render_agent_graph(completed_nodes):
    """Render animated agent graph showing node completion status."""
    st.subheader("🔄 Agent Execution Graph")
    cols = st.columns(len(GRAPH_NODES))
    node_labels = {
        "planner": "📋 Planner",
        "research": "🔍 Research",
        "analytics": "📊 Analytics",
        "critique": "⚖️ Critique",
        "hitl": "👤 HITL",
    }
    for i, node in enumerate(GRAPH_NODES):
        with cols[i]:
            if node in completed_nodes:
                st.success(node_labels[node])
            elif len(completed_nodes) > 0 and GRAPH_NODES[len(completed_nodes)] == node:
                st.warning(f"⏳ {node_labels[node]}")
            else:
                st.info(node_labels[node])


col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📝 Query")
    query = st.text_area(
        "Business Query",
        value="Analyze Apple's revenue performance and strategic priorities from their latest SEC filings",
        height=120,
    )

    example_queries = [
        "Analyze Microsoft's cloud computing growth from recent 10-K filings",
        "What are Tesla's production targets and key risks from latest annual report?",
        "Compare Salesforce and Netflix revenue growth strategies",
        "Summarize Google's AI investments and competitive positioning",
    ]
    selected = st.selectbox(
        "📌 Example queries:", ["Custom query above"] + example_queries
    )
    if selected != "Custom query above":
        query = selected

    if st.button("🚀 Run Agent Graph", type="primary", use_container_width=True):
        st.session_state.nodes_completed = []
        st.session_state.hitl_data = None
        st.session_state.final_output = None
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id

        with st.spinner("Running agent pipeline..."):
            try:
                res = requests.post(
                    f"{API_URL}/run",
                    json={"query": query, "thread_id": thread_id},
                    timeout=120,
                )
                data = res.json()
                st.session_state.nodes_completed = data.get("nodes_completed", [])
                st.session_state.hitl_data = data.get("hitl_interrupt")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.subheader("📊 Agent Output + Scores")
    render_agent_graph(st.session_state.nodes_completed)

    if st.session_state.hitl_data:
        d = st.session_state.hitl_data
        scores = d.get("scores", {})
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
        m2.metric("Coherence", f"{scores.get('coherence', 0):.2f}")
        m3.metric("Task", f"{scores.get('task_completion', 0):.2f}")
        m4.metric("Overall", f"{scores.get('overall', 0):.2f}")
        st.info(f"**Judge Notes:** {d.get('critique_notes', '')}")
        st.text_area(
            "Analytics Result", value=d.get("analytics_result", ""), height=250
        )

    if st.session_state.final_output:
        st.success("✅ Approved Output:")
        st.text_area("Final Output", value=st.session_state.final_output, height=250)

with col3:
    st.subheader("✅ HITL Checkpoint")
    if st.session_state.hitl_data:
        st.warning("⏸️ Agent awaiting your review")
        st.info(
            "Review the analytics output and scores, then approve or request revision."
        )

        if st.button("✅ Approve & Finalize", type="primary", use_container_width=True):
            with st.spinner("Finalizing..."):
                try:
                    res = requests.post(
                        f"{API_URL}/approve",
                        json={
                            "thread_id": st.session_state.thread_id,
                            "action": "approve",
                        },
                        timeout=60,
                    )
                    data = res.json()
                    st.session_state.final_output = data.get("final_output")
                    st.session_state.hitl_data = None
                    st.session_state.nodes_completed.append("hitl")
                    st.success("✅ Approved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        feedback = st.text_area(
            "✏️ Revision feedback:",
            height=100,
            placeholder="e.g. Focus more on Q2 projections and include risk factors",
        )
        if st.button("🔄 Request Revision", use_container_width=True) and feedback:
            with st.spinner("Sending revision..."):
                try:
                    requests.post(
                        f"{API_URL}/approve",
                        json={
                            "thread_id": st.session_state.thread_id,
                            "action": feedback,
                        },
                        timeout=60,
                    )
                    st.info("✅ Revision sent. Re-run query to see updated output.")
                    st.session_state.hitl_data = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    elif not st.session_state.nodes_completed:
        st.info("Run a query to start the agent pipeline.")
    else:
        st.success("Pipeline complete!")

# Footer
st.divider()
st.caption(
    "📚 Knowledge Base: Real SEC EDGAR 10-K/10-Q filings from AAPL, MSFT, GOOGL, CRM, NFLX, TSLA, AMZN, META"
)
st.caption(
    "🏗️ Architecture: LangGraph StateGraph · SQLite Checkpointer · RAGAS + DeepEval Quality Gates · Langfuse Observability"
)
