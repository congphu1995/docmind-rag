"""Tests for agent LLM helper functions."""
from langchain_openai import ChatOpenAI


def test_get_mini_model_returns_chat_openai():
    from backend.app.agent.llm import get_mini_model
    llm = get_mini_model()
    assert isinstance(llm, ChatOpenAI)


def test_get_chat_model_openai():
    from backend.app.agent.llm import get_chat_model
    llm = get_chat_model("openai")
    assert isinstance(llm, ChatOpenAI)


def test_get_chat_model_claude():
    from backend.app.agent.llm import get_chat_model
    from langchain_anthropic import ChatAnthropic
    llm = get_chat_model("claude")
    assert isinstance(llm, ChatAnthropic)


def test_get_chat_model_unknown_raises():
    import pytest
    from backend.app.agent.llm import get_chat_model
    with pytest.raises(ValueError, match="Unknown"):
        get_chat_model("nonexistent")
