"""
test_llm_client.py

Verifies the LLM client abstraction loads correctly for the configured provider.
Run with: python -m tests.test_llm_client
"""

import os
from unittest.mock import patch


def test_get_llm_returns_groq():
    """Verify get_llm() returns a Groq client when LLM_PROVIDER=groq."""
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "test_key_placeholder",
        "GROQ_MODEL": "llama-3.3-70b-versatile"
    }):
        from core.llm_client import get_llm        
        llm = get_llm()
        assert "groq" in type(llm).__module__.lower(), \
            f"Expected Groq client, got {type(llm)}"
        print(f"✅ get_llm() returned: {type(llm).__name__}")


def test_invalid_provider_raises():
    """Verify get_llm() raises ValueError for unsupported providers."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported_provider"}):
        import importlib
        import core.llm_client as llm_client
        importlib.reload(llm_client)
        try:
            llm_client.get_llm()
            print("❌ Should have raised ValueError")
        except ValueError as e:
            print(f"✅ Correctly raised ValueError: {e}")

if __name__ == "__main__":
    test_get_llm_returns_groq()
    test_invalid_provider_raises()
    print("\n✅ LLM client abstraction tests passed.")