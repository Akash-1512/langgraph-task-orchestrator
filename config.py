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
    def __init__(self):
        self.demo_mode         = os.getenv("DEMO_MODE", "true").lower() == "true"
        self.groq_api_key      = os.getenv("GROQ_API_KEY", "")
        self.groq_model        = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.azure_api_key     = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_deployment  = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    @property
    def llm_provider(self) -> str:
        """FREE DEMO: groq | PRODUCTION: set LLM_PROVIDER=azure"""
        return "groq" if self.demo_mode else os.getenv("LLM_PROVIDER", "groq").lower()

    @property
    def vector_store(self) -> str:
        """FREE DEMO: chroma | PRODUCTION: set VECTOR_STORE=qdrant"""
        return "chroma" if self.demo_mode else os.getenv("VECTOR_STORE", "chroma").lower()

    @property
    def checkpoint_db_path(self) -> str:
        """FREE DEMO: SQLite | PRODUCTION: set to PostgreSQL connection string"""
        return os.getenv("CHECKPOINT_DB_PATH", "./checkpoints.sqlite")


settings = Settings()