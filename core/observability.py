"""
core/observability.py — LLM tracing and observability.

FREE DEMO:        Langfuse Cloud (free tier) — cloud.langfuse.com
AZURE PRODUCTION: Azure Application Insights + OpenTelemetry
                  OR Langfuse self-hosted on Azure Container Apps

Gracefully degrades — if keys not set, tracing is disabled.
System continues to work normally without observability.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_callbacks() -> list:
    """
    Returns LangChain callback handlers for the configured observability stack.

    FREE DEMO: Langfuse Cloud — traces every LLM call with latency, tokens, cost
    Sign up free at: https://cloud.langfuse.com

    AZURE PRODUCTION: Azure Application Insights:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    exporter = AzureMonitorTraceExporter(
        connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    )
    # Wire into LangChain via OpenTelemetry callback handler
    # See: https://python.langchain.com/docs/integrations/callbacks/

    Returns:
        List of callback handlers (empty list if observability not configured)
    """
    callbacks = []

    langfuse_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")

    if langfuse_key and langfuse_secret:
        try:
            # Set env vars so Langfuse v3 SDK picks them up automatically
            os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_key
            os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret
            os.environ["LANGFUSE_HOST"] = os.getenv(
                "LANGFUSE_HOST", "https://cloud.langfuse.com"
            )

            from langfuse.langchain import CallbackHandler

            handler = CallbackHandler()  # v3 SDK reads from env vars
            callbacks.append(handler)
        except Exception as e:
            print(f"⚠️  Langfuse init failed: {e} — tracing disabled")

    return callbacks
