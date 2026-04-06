"""
core/mlflow_tracker.py

DagsHub MLflow experiment tracking integration.
Tracks agent runs, scores, and quality metrics.

Free tier: DagsHub provides hosted MLflow at no cost.
Set MLFLOW_TRACKING_URI and DAGSHUB_TOKEN in .env.

Usage:
    from core.mlflow_tracker import log_run
    log_run(query="...", scores={...}, passed=True)
"""

import os

from dotenv import load_dotenv

load_dotenv()


def setup_mlflow():
    """Configure MLflow to use DagsHub as tracking server."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if not tracking_uri or not dagshub_token:
        print("⚠️  MLflow not configured — set MLFLOW_TRACKING_URI and DAGSHUB_TOKEN")
        return False

    try:
        import mlflow

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("langgraph-task-orchestrator")
        return True
    except Exception as e:
        print(f"⚠️  MLflow setup failed: {e}")
        return False


def log_run(
    query: str,
    ragas_scores: dict = None,
    critique_scores: dict = None,
    hitl_approved: bool = None,
    final_output_length: int = None,
) -> None:
    """
    Log a single agent graph run to MLflow.
    Silently skips if MLflow is not configured.

    Args:
        query: The user query that triggered the run
        ragas_scores: Dict with faithfulness, context_recall scores
        critique_scores: Dict with LLM-as-Judge scores
        hitl_approved: Whether the human approved the output
        final_output_length: Character count of final output
    """
    if not setup_mlflow():
        return

    try:
        import mlflow

        with mlflow.start_run():
            mlflow.log_param("query_preview", query[:100])
            mlflow.log_param("query_length", len(query))

            if ragas_scores:
                mlflow.log_metric(
                    "ragas_faithfulness", ragas_scores.get("faithfulness", 0)
                )
                mlflow.log_metric(
                    "ragas_context_recall", ragas_scores.get("context_recall", 0)
                )

            if critique_scores:
                mlflow.log_metric(
                    "critique_faithfulness",
                    critique_scores.get("faithfulness_score", 0),
                )
                mlflow.log_metric(
                    "critique_coherence", critique_scores.get("coherence_score", 0)
                )
                mlflow.log_metric(
                    "critique_overall", critique_scores.get("overall_score", 0)
                )
                mlflow.log_metric(
                    "critique_passed",
                    1 if critique_scores.get("passed_quality_gate") else 0,
                )

            if hitl_approved is not None:
                mlflow.log_metric("hitl_approved", 1 if hitl_approved else 0)

            if final_output_length:
                mlflow.log_metric("final_output_length", final_output_length)

        print("✅ Run logged to MLflow/DagsHub")
    except Exception as e:
        print(f"⚠️  MLflow logging failed: {e}")
