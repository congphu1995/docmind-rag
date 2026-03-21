"""
Langfuse LLM observability for the query pipeline.

Provides a CallbackHandler for LangGraph that auto-traces
all LangChain ChatModel calls (node names, prompts, tokens, cost).

Gracefully disabled when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are empty.
"""
import os

from backend.app.core.config import settings


def is_langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def configure_langfuse() -> None:
    """Set Langfuse env vars so the SDK auto-configures. Call once at startup."""
    if not is_langfuse_enabled():
        return

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)


def get_langfuse_callback():
    """Return a LangChain CallbackHandler for LangGraph, or None if disabled."""
    if not is_langfuse_enabled():
        return None

    from langfuse.langchain import CallbackHandler
    return CallbackHandler()
