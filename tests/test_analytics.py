"""
tests/test_analytics.py

Tests for the Analytics Agent node.
Run with: python -m tests.test_analytics
"""

from langchain_core.documents import Document
from core.retriever import ingest_documents
from agents.state import AgentState
from agents.analytics import analytics_node
import shutil
import os


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
        or splitting into smaller milestones. Underperforming objectives should be analyzed
        for root causes: resource constraints, market changes, or execution gaps.""",
        metadata={"source": "okr_best_practices.pdf", "quarter": "general"}
    ),
]


def test_analytics_node():
    """Verify analytics_node produces a grounded analytical response."""

    # Reset and ingest sample docs
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    ingest_documents(SAMPLE_OKR_DOCUMENTS)

    state: AgentState = {
        "query": "Analyze Q1 OKR performance and suggest adjustments for Q2",
        "messages": [],
        "plan": [
            "Retrieve Q1 OKR performance data from the knowledge base",
            "Calculate key result completion rates for each objective",
            "Identify underperforming objectives (below 70% completion)",
            "Analyze root causes for underperformance using retrieved context",
            "Suggest specific, measurable OKR adjustments for Q2",
        ],
        "research_context": [doc.page_content for doc in SAMPLE_OKR_DOCUMENTS],
        "retrieved_sources": [doc.metadata["source"] for doc in SAMPLE_OKR_DOCUMENTS],
        "analytics_result": None,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
    }

    result = analytics_node(state)

    assert "analytics_result" in result
    assert isinstance(result["analytics_result"], str)
    assert len(result["analytics_result"]) > 100

    print("✅ Analytics Agent produced response:")
    print("-" * 60)
    print(result["analytics_result"])
    print("-" * 60)


if __name__ == "__main__":
    test_analytics_node()
    print("\n✅ Analytics agent tests passed.")