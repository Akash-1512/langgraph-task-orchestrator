"""
config.py — Central configuration with DEMO_MODE support.

DEMO_MODE=true  → Forces all providers to free-tier alternatives.
                  No credit card, no Azure subscription required.
DEMO_MODE=false → Uses environment-configured providers (Azure, Qdrant, etc.)

Usage:
    from config import settings
    if settings.demo_mode:
        ...
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    demo_mode: bool = os.getenv("DEMO_MODE", "true").lower() == "true"

    # LLM
    @property
    def llm_provider(self) -> str:
        if self.demo_mode:
            return "groq"
        return os.getenv("LLM_PROVIDER", "groq").lower()

    # Vector store
    @property
    def vector_store(self) -> str:
        if self.demo_mode:
            return "chroma"
        return os.getenv("VECTOR_STORE", "chroma").lower()

    # Checkpoint
    @property
    def checkpoint_db_path(self) -> str:
        return os.getenv("CHECKPOINT_DB_PATH", "./checkpoints.sqlite")

    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Azure OpenAI
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")


settings = Settings()