from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.agent.nodes.decomposer import decomposer


def _make_state(query: str) -> dict:
    return {
        "original_query": query,
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": "multi_hop",
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


@patch("backend.app.agent.nodes.decomposer.get_mini_model")
async def test_decomposer_splits_into_sub_questions(mock_get_mini):
    mock_llm = MagicMock()
    mock_get_mini.return_value = mock_llm

    mock_bound = AsyncMock()
    mock_llm.bind.return_value = mock_bound
    mock_bound.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="What was the revenue in Q1?\nWhat was the revenue in Q2?\nHow do they compare?"
        )
    )

    state = _make_state("How did Q1 revenue compare to Q2?")
    result = await decomposer(state)

    assert len(result["sub_questions"]) == 3
    assert "revenue" in result["sub_questions"][0].lower()


@patch("backend.app.agent.nodes.decomposer.get_mini_model")
async def test_decomposer_fallback_on_empty(mock_get_mini):
    mock_llm = MagicMock()
    mock_get_mini.return_value = mock_llm

    mock_bound = AsyncMock()
    mock_llm.bind.return_value = mock_bound
    mock_bound.ainvoke = AsyncMock(return_value=MagicMock(content=""))

    state = _make_state("Complex question here")
    result = await decomposer(state)

    assert len(result["sub_questions"]) == 1
    assert result["sub_questions"][0] == "Complex question here"
