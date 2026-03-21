from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.agent.nodes.query_analyzer import query_analyzer
from backend.app.schemas.chat import QueryAnalysis


def _make_state(query: str) -> dict:
    return {
        "original_query": query,
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": "",
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "",
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


@patch("backend.app.agent.nodes.query_analyzer.get_mini_model")
async def test_factual_query_classified(mock_get_mini):
    mock_llm = MagicMock()
    mock_get_mini.return_value = mock_llm

    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.ainvoke = AsyncMock(
        return_value=QueryAnalysis(
            query_type="factual",
            language="en",
            sub_questions=[],
            filters={},
        )
    )

    state = _make_state("What is the revenue in Q1 2024?")
    result = await query_analyzer(state)

    assert result["query_type"] == "factual"
    assert result["detected_language"] == "en"
    assert len(result["agent_trace"]) > 0


@patch("backend.app.agent.nodes.query_analyzer.get_mini_model")
async def test_greeting_classified(mock_get_mini):
    mock_llm = MagicMock()
    mock_get_mini.return_value = mock_llm

    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.ainvoke = AsyncMock(
        return_value=QueryAnalysis(query_type="greeting", language="en")
    )

    state = _make_state("Hello!")
    result = await query_analyzer(state)

    assert result["query_type"] == "greeting"


@patch("backend.app.agent.nodes.query_analyzer.get_mini_model")
async def test_analysis_failure_defaults_to_factual(mock_get_mini):
    mock_llm = MagicMock()
    mock_get_mini.return_value = mock_llm

    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    state = _make_state("What is the policy limit?")
    result = await query_analyzer(state)

    assert result["query_type"] == "factual"
    assert result["detected_language"] == "en"
