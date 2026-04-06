"""
core/llm_client.py — Provider-agnostic LLM abstraction layer.

FREE DEMO:        LLM_PROVIDER=groq       → Groq API (free tier)
AZURE PRODUCTION: LLM_PROVIDER=azure      → Azure OpenAI GPT-4o
OTHER OPTIONS:    LLM_PROVIDER=openai     → OpenAI direct
                  LLM_PROVIDER=anthropic  → Anthropic Claude

Change ONE env variable to swap providers. Zero code changes.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.1):
    """
    Returns a LangChain-compatible LLM for the configured provider.

    FREE DEMO: Uses Groq llama-3.3-70b-versatile (free tier, no card)
    AZURE PRODUCTION: Uses AzureChatOpenAI with GPT-4o deployment

    Args:
        temperature: Controls output randomness (0.0 = deterministic)

    Returns:
        LangChain BaseChatModel instance
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        # FREE DEMO: Groq API — free tier, 100K tokens/day
        # Sign up at: https://console.groq.com
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
        # AZURE PRODUCTION: Replace above block with:
        # from langchain_openai import AzureChatOpenAI
        # return AzureChatOpenAI(
        #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        #     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        #     temperature=temperature,
        # )
        # AZURE SETUP: az cognitiveservices account create --kind OpenAI
        #              az cognitiveservices account deployment create --name gpt-4o

    elif provider == "azure":
        # AZURE PRODUCTION: Azure OpenAI GPT-4o
        # Requires: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
        #           AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=temperature,
        )
        # AZURE KEY VAULT: Replace direct env vars with:
        # from azure.keyvault.secrets import SecretClient
        # from azure.identity import DefaultAzureCredential
        # client = SecretClient(vault_url=os.getenv("AZURE_KEY_VAULT_URL"),
        #                       credential=DefaultAzureCredential())
        # api_key = client.get_secret("azure-openai-api-key").value

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514",
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Valid options: groq | azure | openai | anthropic"
        )
