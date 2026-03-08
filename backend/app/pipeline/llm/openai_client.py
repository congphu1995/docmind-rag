import time
from typing import AsyncGenerator

from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.metrics import LLM_REQUEST_DURATION, LLM_TOKENS_TOTAL
from backend.app.pipeline.base.llm_client import BaseLLMClient


class OpenAIClient(BaseLLMClient):

    def __init__(self, model: str = "gpt-4o"):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
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
        msgs = self._build_messages(messages, system)
        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        duration = time.perf_counter() - start
        LLM_REQUEST_DURATION.labels(provider="openai", model=self._model).observe(
            duration
        )
        if hasattr(response, "usage") and response.usage:
            LLM_TOKENS_TOTAL.labels(
                provider="openai", model=self._model, type="prompt"
            ).inc(response.usage.prompt_tokens)
            LLM_TOKENS_TOTAL.labels(
                provider="openai", model=self._model, type="completion"
            ).inc(response.usage.completion_tokens)
        return response.choices[0].message.content

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        msgs = self._build_messages(messages, system)
        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 4096),
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        duration = time.perf_counter() - start
        LLM_REQUEST_DURATION.labels(provider="openai", model=self._model).observe(
            duration
        )

    async def complete_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        system: str | None = None,
        **kwargs,
    ) -> BaseModel:
        """OpenAI structured output — uses .parse() for reliable JSON."""
        msgs = self._build_messages(messages, system)
        start = time.perf_counter()
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=msgs,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 200),
            response_format=response_model,
        )
        duration = time.perf_counter() - start
        LLM_REQUEST_DURATION.labels(provider="openai", model=self._model).observe(
            duration
        )
        if hasattr(response, "usage") and response.usage:
            LLM_TOKENS_TOTAL.labels(
                provider="openai", model=self._model, type="prompt"
            ).inc(response.usage.prompt_tokens)
            LLM_TOKENS_TOTAL.labels(
                provider="openai", model=self._model, type="completion"
            ).inc(response.usage.completion_tokens)
        return response.choices[0].message.parsed

    def _build_messages(
        self, messages: list[dict], system: str | None
    ) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        return msgs
