import time
from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from backend.app.core.config import settings
from backend.app.core.metrics import LLM_REQUEST_DURATION, LLM_TOKENS_TOTAL
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
        start = time.perf_counter()
        response = await self._client.messages.create(
            model=self._model,
            system=system or "",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
        )
        duration = time.perf_counter() - start
        LLM_REQUEST_DURATION.labels(
            provider="anthropic", model=self._model
        ).observe(duration)
        if hasattr(response, "usage") and response.usage:
            LLM_TOKENS_TOTAL.labels(
                provider="anthropic", model=self._model, type="prompt"
            ).inc(response.usage.input_tokens)
            LLM_TOKENS_TOTAL.labels(
                provider="anthropic", model=self._model, type="completion"
            ).inc(response.usage.output_tokens)
        return response.content[0].text

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        start = time.perf_counter()
        async with self._client.messages.stream(
            model=self._model,
            system=system or "",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
        ) as stream:
            async for text in stream.text_stream:
                yield text
        duration = time.perf_counter() - start
        LLM_REQUEST_DURATION.labels(
            provider="anthropic", model=self._model
        ).observe(duration)
