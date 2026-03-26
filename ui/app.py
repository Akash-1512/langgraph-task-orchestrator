"""
ui/app.py — Streamlit frontend for langgraph-task-orchestrator
Run with: streamlit run ui/app.py
"""

import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"

st.set_page_config(page_title="OKR Analytics Agent", layout="wide")
st.title("🤖 Multi-Agent OKR Analytics System")
st.caption("Powered by LangGraph · Groq · ChromaDB")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "hitl_data" not in st.session_state:
    st.session_state.hitl_data = None

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📝 Query")
    query = st.text_area("Business Query", value="Analyze Q1 OKR performance and suggest adjustments for Q2", height=120)
    if st.button("🚀 Run Agent Graph", type="primary"):
        with st.spinner("Running agent pipeline..."):
            res = requests.post(f"{API_URL}/run", json={"query": query, "thread_id": str(uuid.uuid4())})
            data = res.json()
            st.session_state.thread_id = data["thread_id"]
            st.session_state.hitl_data = data.get("hitl_interrupt")
            st.success(f"Nodes completed: {', '.join(data['nodes_completed'])}")

with col2:
    st.subheader("📊 Agent Output + Scores")
    if st.session_state.hitl_data:
        d = st.session_state.hitl_data
        scores = d.get("scores", {})
        st.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
        st.metric("Coherence", f"{scores.get('coherence', 0):.2f}")
        st.metric("Task Completion", f"{scores.get('task_completion', 0):.2f}")
        st.metric("Overall", f"{scores.get('overall', 0):.2f}")
        st.info(f"**Critique:** {d.get('critique_notes', '')}")
        st.text_area("Analytics Result", value=d.get("analytics_result", ""), height=300)

with col3:
    st.subheader("✅ HITL Checkpoint")
    if st.session_state.hitl_data:
        st.warning("⏸️ Agent awaiting your approval")
        if st.button("✅ Approve", type="primary"):
            with st.spinner("Resuming graph..."):
                res = requests.post(f"{API_URL}/approve", json={"thread_id": st.session_state.thread_id, "action": "approve"})
                data = res.json()
                st.success("✅ Approved! Final output ready.")
                st.text_area("Final Output", value=data.get("final_output", ""), height=300)
                st.session_state.hitl_data = None

        feedback = st.text_area("Or provide revision feedback:")
        if st.button("🔄 Request Revision") and feedback:
            with st.spinner("Sending revision..."):
                res = requests.post(f"{API_URL}/approve", json={"thread_id": st.session_state.thread_id, "action": feedback})
                st.info("Revision submitted. Re-run to see updated output.")
                st.session_state.hitl_data = None