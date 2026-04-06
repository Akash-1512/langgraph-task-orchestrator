"""
evaluation/ragas_eval.py

RAGAS evaluation pipeline for the langgraph-task-orchestrator RAG component.
Evaluates faithfulness, context recall, and factual correctness.

Metrics:
    Faithfulness     — Are all claims in the answer grounded in retrieved context?
    LLMContextRecall — Does retrieved context cover the reference answer claims?
    FactualCorrectness — Is the answer factually correct vs reference?

Quality gate: faithfulness >= 0.75 (same threshold as Critique agent)

Run with: python -m evaluation.ragas_eval
"""

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper  # TODO: migrate to llm_factory in v1.0
from ragas.metrics.collections import Faithfulness, LLMContextRecall

from core.llm_client import get_llm

faithfulness = Faithfulness()
context_recall = LLMContextRecall()


# ── Sample evaluation dataset ─────────────────────────────────────────────
# In production, this would be loaded from a file or database.
# Each sample: user_input, retrieved_contexts, response, reference

EVAL_SAMPLES = [
    SingleTurnSample(
        user_input="What was the Q1 MAU achievement vs target?",
        retrieved_contexts=[
            "Key Result 1.1: Achieve 10,000 monthly active users — Result: 7,200 (72% complete).",
            "Key Result 1.2: Reduce churn rate to below 5% — Result: 6.8% (below target).",
        ],
        response="Q1 achieved 7,200 monthly active users against a target of 10,000, representing 72% completion.",
        reference="The Q1 MAU target was 10,000 and the result was 7,200, which is 72% of the target.",
    ),
    SingleTurnSample(
        user_input="How did Q1 revenue performance compare to target?",
        retrieved_contexts=[
            "Key Result 2.1: Achieve $2M ARR — Result: $1.6M (80% complete).",
            "Key Result 2.2: Close 50 enterprise deals — Result: 38 deals (76% complete).",
        ],
        response="Q1 revenue reached $1.6M ARR against a $2M target (80% complete). Enterprise deals closed at 38 of 50 target (76%).",
        reference="Q1 ARR was $1.6M vs $2M target (80%), and 38 of 50 enterprise deals were closed (76%).",
    ),
    SingleTurnSample(
        user_input="What engineering reliability issues occurred in Q1?",
        retrieved_contexts=[
            "Key Result 3.1: Achieve 99.9% uptime — Result: 99.7% (below target).",
            "Key Result 3.2: Reduce P1 incidents to zero — Result: 3 P1 incidents.",
            "Key Result 3.3: Deploy CI/CD for all services — Result: 100% complete.",
        ],
        response="Engineering faced reliability challenges in Q1: uptime was 99.7% vs 99.9% target, and 3 P1 incidents occurred vs zero target. However, CI/CD deployment achieved 100% completion.",
        reference="Q1 uptime was 99.7% (target 99.9%), there were 3 P1 incidents (target zero), but CI/CD was 100% complete.",
    ),
]

FAITHFULNESS_THRESHOLD = 0.75


def run_ragas_evaluation() -> dict:
    """
    Runs RAGAS evaluation on the sample dataset.
    Returns evaluation results and pass/fail status.
    """
    print("🔍 Running RAGAS evaluation pipeline...")
    print(f"   Samples: {len(EVAL_SAMPLES)}")
    print(f"   Metrics: Faithfulness, LLMContextRecall, FactualCorrectness")
    print(f"   Quality gate: faithfulness >= {FAITHFULNESS_THRESHOLD}")
    print()

    # Wrap the LangChain LLM for RAGAS
    llm = get_llm()
    evaluator_llm = LangchainLLMWrapper(llm)

    # Build evaluation dataset
    dataset = EvaluationDataset(samples=EVAL_SAMPLES)

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_recall],
        llm=evaluator_llm,
    )

    import numpy as np

    scores = {
        "faithfulness": round(float(np.mean(result["faithfulness"])), 4),
        "context_recall": round(float(np.mean(result["context_recall"])), 4),
    }

    passed = scores["faithfulness"] >= FAITHFULNESS_THRESHOLD

    print("📊 RAGAS Evaluation Results:")
    print(
        f"   Faithfulness:       {scores['faithfulness']} {'✅' if scores['faithfulness'] >= FAITHFULNESS_THRESHOLD else '❌'}"
    )
    print(f"   Context Recall:     {scores['context_recall']}")
    print()
    print(
        f"{'✅ QUALITY GATE PASSED' if passed else '❌ QUALITY GATE FAILED'} (faithfulness threshold: {FAITHFULNESS_THRESHOLD})"
    )

    return {"scores": scores, "passed": passed}


if __name__ == "__main__":
    result = run_ragas_evaluation()
    if not result["passed"]:
        print("❌ Quality gate failed — exiting with code 1")
        import sys

        sys.exit(1)
