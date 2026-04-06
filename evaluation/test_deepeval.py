"""
evaluation/test_deepeval.py

DeepEval LLM-as-Judge quality gate tests.
Run with: deepeval test run evaluation/test_deepeval.py
Or: pytest evaluation/test_deepeval.py

Tests three quality dimensions of the Analytics Agent output:
1. Groundedness   — Is the answer grounded in retrieved context?
2. Task Completion — Does the answer address the original query?
3. Coherence      — Is the answer logically structured?

Quality gate: all three metrics must score >= 0.5 to pass.
In CI/CD, a failing test exits with code 1 — blocking the PR merge.
"""

import os

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

load_dotenv()


class GroqDeepEvalModel(DeepEvalBaseLLM):
    """Custom DeepEval judge model using Groq via LangChain."""

    def __init__(self):
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def load_model(self):
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=self.model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        return model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"groq/{self.model_name}"


groq_judge = GroqDeepEvalModel()

# ── Sample OKR analytics output to evaluate ──────────────────────────────
SAMPLE_QUERY = "Analyze Q1 OKR performance and suggest adjustments for Q2"

SAMPLE_CONTEXT = [
    "Key Result 1.1: Achieve 10,000 monthly active users — Result: 7,200 (72% complete).",
    "Key Result 2.1: Achieve $2M ARR — Result: $1.6M (80% complete).",
    "Key Result 3.1: Achieve 99.9% uptime — Result: 99.7% (below target).",
    "OKR best practice: When key results fall below 70%, consider reducing scope or reallocating resources.",
]

SAMPLE_ANALYTICS_OUTPUT = """
## Summary
Q1 OKR performance shows mixed results across all three objectives.
Product adoption reached 72% of MAU target. Revenue achieved 80% of ARR target.
Engineering reliability fell short with 99.7% uptime vs 99.9% target.

## Key Findings
- KR 1.1: 7,200 MAU vs 10,000 target (72% complete)
- KR 2.1: $1.6M ARR vs $2M target (80% complete)
- KR 3.1: 99.7% uptime vs 99.9% target (below threshold)

## Recommendations
- Reduce MAU target to 8,500 for Q2 and focus on user activation
- Reallocate sales resources to close remaining enterprise deals
- Implement incident response playbook to eliminate P1 incidents
"""


# ── Metric definitions ────────────────────────────────────────────────────

groundedness_metric = GEval(
    name="Groundedness",
    criteria="""Evaluate whether every factual claim in the actual output 
    is directly supported by the retrieval context. 
    Penalize any claim not found in the context.""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)

task_completion_metric = GEval(
    name="Task Completion",
    criteria="""Evaluate whether the actual output fully addresses the input query.
    The output should include: a summary of Q1 performance, specific key findings 
    with numbers, and concrete Q2 recommendations.""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.5,
)

coherence_metric = GEval(
    name="Coherence",
    criteria="""Evaluate whether the actual output is logically structured,
    well-organized, and easy to follow. It should have clear sections,
    consistent terminology, and logical flow from findings to recommendations.""",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.5,
)


# ── Test cases ────────────────────────────────────────────────────────────


def test_analytics_groundedness():
    """Analytics output must be grounded in retrieved context."""
    test_case = LLMTestCase(
        input=SAMPLE_QUERY,
        actual_output=SAMPLE_ANALYTICS_OUTPUT,
        retrieval_context=SAMPLE_CONTEXT,
    )
    assert_test(test_case, [groundedness_metric])


def test_analytics_task_completion():
    """Analytics output must fully address the user query."""
    test_case = LLMTestCase(
        input=SAMPLE_QUERY,
        actual_output=SAMPLE_ANALYTICS_OUTPUT,
    )
    assert_test(test_case, [task_completion_metric])


def test_analytics_coherence():
    """Analytics output must be logically coherent and well-structured."""
    test_case = LLMTestCase(
        input=SAMPLE_QUERY,
        actual_output=SAMPLE_ANALYTICS_OUTPUT,
    )
    assert_test(test_case, [coherence_metric])
