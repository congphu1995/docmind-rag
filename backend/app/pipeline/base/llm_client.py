from abc import ABC, abstractmethod
from typing import AsyncGenerator

from pydantic import BaseModel


class BaseLLMClient(ABC):
    """
    Abstract base — all LLM providers implement this interface.
    Services import BaseLLMClient only, never a concrete class.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> str:
        """Single-turn completion. Returns full text response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Streaming completion. Yields text chunks."""
        ...

    async def complete_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        system: str | None = None,
        **kwargs,
    ) -> BaseModel:
        """
        Structured output — returns validated Pydantic model.
        Default: parse JSON from text. OpenAIClient overrides with .parse().
        """
        text = await self.complete(messages, system, **kwargs)
        return response_model.model_validate_json(text)

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
