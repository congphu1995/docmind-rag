"""
LangChain ChatModel factories for agent nodes.

Agent nodes use LangChain models (not raw SDKs) so that
LangGraph's CallbackHandler auto-traces all LLM calls.

The ingestion pipeline keeps using LLMFactory + raw SDKs.
"""
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from backend.app.core.config import settings


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    streaming: bool = False,
):
    """Create a LangChain ChatModel for the given provider."""
    provider = provider or settings.default_llm

    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=settings.openai_api_key,
            streaming=streaming,
        )
    elif provider == "claude":
        return ChatAnthropic(
            model_name=model or "claude-sonnet-4-20250514",
            api_key=settings.anthropic_api_key,
            streaming=streaming,
        )

    raise ValueError(f"Unknown LLM provider: {provider}. Choose: openai, claude")


def get_mini_model():
    """GPT-4o-mini for query analysis, rewriting, HyDE, decomposition."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
    )
