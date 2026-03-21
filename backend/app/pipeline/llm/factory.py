from backend.app.core.config import settings
from backend.app.pipeline.base.llm_client import BaseLLMClient


class LLMFactory:
    @staticmethod
    def create(provider: str | None = None, model: str | None = None) -> BaseLLMClient:
        """Create LLM client by provider name."""
        provider = provider or settings.default_llm

        if provider == "openai":
            from backend.app.pipeline.llm.openai_client import OpenAIClient

            return OpenAIClient(model=model or "gpt-4o")
        elif provider == "claude":
            from backend.app.pipeline.llm.claude_client import ClaudeClient

            return ClaudeClient(model=model or "claude-sonnet-4-20250514")

        raise ValueError(f"Unknown LLM provider: {provider}. Choose: openai, claude")

    @staticmethod
    def create_mini() -> BaseLLMClient:
        """GPT-4o-mini for HyDE, metadata extraction, enrichment."""
        from backend.app.pipeline.llm.openai_client import OpenAIClient

        return OpenAIClient(model="gpt-4o-mini")
