from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from backend.app.core.config import settings
from backend.app.pipeline.base.llm_client import BaseLLMClient


class ClaudeClient(BaseLLMClient):

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> str:
        response = await self._client.messages.create(
            model=self._model,
            system=system or "",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
        )
        return response.content[0].text

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        async with self._client.messages.stream(
            model=self._model,
            system=system or "",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
        ) as stream:
            async for text in stream.text_stream:
                yield text
