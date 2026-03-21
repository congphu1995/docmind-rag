from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.agent.nodes.query_rewriter import query_rewriter, _should_use_hyde


def _make_state(query: str, query_type: str = "factual") -> dict:
    return {
        "original_query": query,
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": query_type,
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": "",
        "hyde_query": "",
        "hyde_used": False,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "retrieval_attempts": 0,
        "retrieval_quality": 0.0,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


def test_hyde_skipped_for_greeting():
    assert _should_use_hyde("Hello!", "greeting") is False


def test_hyde_skipped_for_general():
    assert _should_use_hyde("What is the weather?", "general") is False


def test_hyde_skipped_for_factual():
    assert _should_use_hyde("What is the revenue?", "factual") is False


def test_hyde_used_for_analytical():
    assert _should_use_hyde("How does revenue compare?", "analytical") is True


def test_hyde_used_for_multi_hop():
    assert _should_use_hyde("Impact of policy on claims?", "multi_hop") is True


@patch("backend.app.agent.nodes.query_rewriter.get_mini_model")
async def test_rewriter_with_hyde(mock_get_mini):
    mock_llm = MagicMock()
    mock_get_mini.return_value = mock_llm

    mock_bound = AsyncMock()
    mock_llm.bind.return_value = mock_bound
    mock_bound.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content="expanded query about revenue comparison"),
            MagicMock(content="hypothetical answer about revenue"),
        ]
    )

    state = _make_state("How does the revenue compare between Q1 and Q2?", "analytical")
    result = await query_rewriter(state)

    assert result["hyde_used"] is True
    assert result["hyde_query"] != ""
    assert result["rewritten_query"] != ""


@patch("backend.app.agent.nodes.query_rewriter.get_mini_model")
async def test_rewriter_without_hyde(mock_get_mini):
    mock_llm = MagicMock()
    mock_get_mini.return_value = mock_llm

    mock_bound = AsyncMock()
    mock_llm.bind.return_value = mock_bound
    mock_bound.ainvoke = AsyncMock(return_value=MagicMock(content="revenue Q1"))

    state = _make_state("Revenue Q1", "factual")
    result = await query_rewriter(state)

    assert result["hyde_used"] is False
    assert result["hyde_query"] == ""
