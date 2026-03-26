"""
tests/test_graph.py

Integration test for the full LangGraph StateGraph.
Tests the complete pipeline from query to HITL approval.
Run with: python -m tests.test_graph
"""

import shutil
import os
from langchain_core.documents import Document
from langgraph.types import Command

from graph.agent_graph import graph
from core.retriever import ingest_documents


SAMPLE_OKR_DOCUMENTS = [
    Document(
        page_content="""Q1 2024 OKR Performance Report.
        Objective 1: Increase product adoption.
        Key Result 1.1: Achieve 10,000 monthly active users — Result: 7,200 (72% complete).
        Key Result 1.2: Reduce churn rate to below 5% — Result: 6.8% (below target).
        Key Result 1.3: Launch 3 new features — Result: 2 launched (67% complete).""",
        metadata={"source": "q1_okr_report.pdf", "quarter": "Q1 2024"}
    ),
    Document(
        page_content="""Q1 2024 Sales OKR Performance.
        Objective 2: Grow revenue.
        Key Result 2.1: Achieve $2M ARR — Result: $1.6M (80% complete).
        Key Result 2.2: Close 50 enterprise deals — Result: 38 deals (76% complete).
        Key Result 2.3: Expand to 3 new markets — Result: 2 markets (67% complete).""",
        metadata={"source": "q1_sales_okr.pdf", "quarter": "Q1 2024"}
    ),
    Document(
        page_content="""Q1 2024 Engineering OKR Performance.
        Objective 3: Improve platform reliability.
        Key Result 3.1: Achieve 99.9% uptime — Result: 99.7% (below target).
        Key Result 3.2: Reduce P1 incidents to zero — Result: 3 P1 incidents.
        Key Result 3.3: Deploy CI/CD for all services — Result: 100% complete.""",
        metadata={"source": "q1_engineering_okr.pdf", "quarter": "Q1 2024"}
    ),
    Document(
        page_content="""OKR Best Practices for Q2 Adjustments.
        When key results fall below 70%, consider: reducing scope, reallocating resources,
        or splitting into smaller milestones.""",
        metadata={"source": "okr_best_practices.pdf", "quarter": "general"}
    ),
]


def test_full_graph_with_hitl_approval():
    """
    Integration test: runs the full graph and simulates human approval.
    """
    # Reset vector store
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    ingest_documents(SAMPLE_OKR_DOCUMENTS)

    # Thread config — required for checkpointer to track state across interrupts
    config = {"configurable": {"thread_id": "test-thread-001"}}

    initial_input = {
        "query": "Analyze Q1 OKR performance and suggest adjustments for Q2",
        "messages": [],
        "plan": None,
        "research_context": None,
        "retrieved_sources": None,
        "analytics_result": None,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
    }

    print("\n🚀 Starting graph execution...")
    print("=" * 60)

    # Run graph until interrupt
    for event in graph.stream(initial_input, config=config):
        node_name = list(event.keys())[0]
        if node_name == "__interrupt__":
            interrupt_data = event["__interrupt__"][0].value
            print(f"\n⏸️  HITL Interrupt triggered.")
            print(f"   Overall Score: {interrupt_data['scores']['overall']}")
            print(f"   Notes: {interrupt_data['critique_notes']}")
            print(f"   Instructions: {interrupt_data['instructions']}")
        else:
            print(f"✅ Node completed: {node_name}")

    # Simulate human approval
    print("\n👤 Human reviewer: approve")
    for event in graph.stream(Command(resume="approve"), config=config):
        node_name = list(event.keys())[0]
        print(f"✅ Node completed: {node_name}")

    # Verify final state
    final_state = graph.get_state(config)
    assert final_state.values.get("hitl_status") == "approved"
    assert final_state.values.get("final_output") is not None

    print("\n" + "=" * 60)
    print("✅ Graph integration test passed.")
    print(f"   HITL Status: {final_state.values.get('hitl_status')}")
    print(f"   Final output length: {len(final_state.values.get('final_output', ''))} chars")


if __name__ == "__main__":
    test_full_graph_with_hitl_approval()