"""
Smoke tests for Langfuse integration.

Verifies that the wiring between Langfuse, LangChain ChatModels,
and LangGraph is correct without requiring live Langfuse credentials.
"""

import inspect
from unittest.mock import patch

from backend.app.agent.graph import build_graph
from backend.app.agent.llm import get_chat_model, get_mini_model
from backend.app.agent.nodes.decomposer import decomposer
from backend.app.agent.nodes.generator import (
    direct_llm,
    direct_response,
    generator_node,
)
from backend.app.agent.nodes.query_analyzer import query_analyzer
from backend.app.agent.nodes.query_rewriter import query_rewriter
from backend.app.agent.nodes.retriever import retriever_node
from backend.app.core.langfuse import get_langfuse_callback, is_langfuse_enabled


def test_callback_disabled_when_keys_empty():
    """get_langfuse_callback returns None when keys are not set."""
    with patch("backend.app.core.langfuse.settings") as mock_settings:
        mock_settings.langfuse_public_key = ""
        mock_settings.langfuse_secret_key = ""
        assert get_langfuse_callback() is None
        assert not is_langfuse_enabled()


def test_graph_compiles_with_all_nodes():
    """Graph still compiles after refactoring nodes to LangChain."""
    graph = build_graph()
    assert graph is not None
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "query_analyzer",
        "query_rewriter",
        "decomposer",
        "retriever",
        "reranker",
        "generator",
        "direct_response",
        "direct_llm",
    }
    assert expected.issubset(node_names)


def test_agent_nodes_accept_config_param():
    """All agent node functions accept an optional config parameter."""
    nodes = [
        query_analyzer,
        query_rewriter,
        decomposer,
        retriever_node,
        generator_node,
        direct_response,
        direct_llm,
    ]
    for node_fn in nodes:
        sig = inspect.signature(node_fn)
        assert "config" in sig.parameters, (
            f"{node_fn.__name__} missing 'config' parameter"
        )
        param = sig.parameters["config"]
        assert param.default is None, (
            f"{node_fn.__name__} 'config' should default to None"
        )


@patch("backend.app.agent.llm.settings")
def test_get_chat_model_returns_langchain_model(mock_settings):
    """get_chat_model returns LangChain ChatOpenAI or ChatAnthropic."""
    mock_settings.default_llm = "openai"
    mock_settings.openai_api_key = "test-key"

    model = get_chat_model("openai")
    assert hasattr(model, "ainvoke"), "Model must have ainvoke for async usage"
    assert hasattr(model, "astream"), "Model must have astream for streaming"
    assert hasattr(model, "with_structured_output"), (
        "Model must support structured output"
    )


@patch("backend.app.agent.llm.settings")
def test_get_mini_model_returns_langchain_model(mock_settings):
    """get_mini_model returns a LangChain ChatOpenAI."""
    mock_settings.openai_api_key = "test-key"

    model = get_mini_model()
    assert hasattr(model, "ainvoke")
    assert hasattr(model, "bind")
    assert hasattr(model, "with_structured_output")
