"""
llm_client.py

Provider-agnostic LLM abstraction layer.
Returns a LangChain-compatible chat model based on LLM_PROVIDER env variable.

Supported providers:
    groq    → Groq API (free tier, llama-3.3-70b-versatile) — default for demo
    azure   → Azure OpenAI (gpt-4o) — production config
    openai  → OpenAI API (gpt-4o) — alternative production config
    anthropic → Anthropic Claude — alternative production config

Usage:
    from llm_client import get_llm
    llm = get_llm()

Swap provider by changing LLM_PROVIDER in .env — zero code changes required.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """
    Returns a LangChain-compatible chat model for the configured provider.
    Reads LLM_PROVIDER from environment. Defaults to 'groq' if not set.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )

    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.1,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            f"Supported values: groq, azure, openai, anthropic"
        )