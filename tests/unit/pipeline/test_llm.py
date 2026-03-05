import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from backend.app.pipeline.llm.factory import LLMFactory


def test_factory_creates_openai():
    with patch("backend.app.pipeline.llm.openai_client.AsyncOpenAI"):
        client = LLMFactory.create("openai")
        from backend.app.pipeline.llm.openai_client import OpenAIClient
        assert isinstance(client, OpenAIClient)


def test_factory_creates_claude():
    with patch("backend.app.pipeline.llm.claude_client.AsyncAnthropic"):
        client = LLMFactory.create("claude")
        from backend.app.pipeline.llm.claude_client import ClaudeClient
        assert isinstance(client, ClaudeClient)


def test_factory_creates_mini():
    with patch("backend.app.pipeline.llm.openai_client.AsyncOpenAI"):
        client = LLMFactory.create_mini()
        assert client.model_name == "gpt-4o-mini"


def test_factory_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create("nonexistent")


def test_factory_default_models():
    with patch("backend.app.pipeline.llm.openai_client.AsyncOpenAI"):
        client = LLMFactory.create("openai")
        assert client.model_name == "gpt-4o"

    with patch("backend.app.pipeline.llm.claude_client.AsyncAnthropic"):
        client = LLMFactory.create("claude")
        assert "claude" in client.model_name


async def test_openai_complete():
    with patch("backend.app.pipeline.llm.openai_client.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        from backend.app.pipeline.llm.openai_client import OpenAIClient
        client = OpenAIClient()
        result = await client.complete([{"role": "user", "content": "Hi"}])
        assert result == "Hello!"


async def test_claude_complete():
    with patch("backend.app.pipeline.llm.claude_client.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        from backend.app.pipeline.llm.claude_client import ClaudeClient
        client = ClaudeClient()
        result = await client.complete([{"role": "user", "content": "Hi"}])
        assert result == "Hello from Claude!"
