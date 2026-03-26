"""
core/observability.py

Langfuse LLM observability integration (SDK v3).
Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from .env.
If keys are not set, observability is silently disabled — no crashes.

Usage:
    from core.observability import get_callbacks
    callbacks = get_callbacks()
    llm.invoke(messages, config={"callbacks": callbacks})
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_callbacks() -> list:
    """
    Returns a list of active Langfuse callbacks for LangChain invocations.
    Empty list if Langfuse keys are not configured — safe for any invoke call.

    Usage:
        callbacks = get_callbacks()
        llm.invoke(messages, config={"callbacks": callbacks})
        graph.stream(input, config={"callbacks": callbacks})
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return []

    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
        return [handler]
    except Exception as e:
        print(f"⚠️  Langfuse init failed: {e} — observability disabled.")
        return []