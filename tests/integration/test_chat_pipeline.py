"""
Requires: Qdrant + PostgreSQL running (docker compose up qdrant postgres)
Requires: OPENAI_API_KEY set (for embeddings + gpt-4o-mini)
Run with: pytest tests/integration/test_chat_pipeline.py -m integration
"""
import pytest

from backend.app.schemas.chat import ChatRequest


@pytest.mark.integration
async def test_chat_non_streaming(sample_pdf_path):
    """Ingest a document, then query it non-streaming."""
    from backend.app.services.ingestion import IngestionService
    from backend.app.services.rag import RAGService

    # Ingest
    ingestion = IngestionService()
    ingest_result = await ingestion.ingest(sample_pdf_path, "sample.pdf")
    doc_id = ingest_result["doc_id"]

    # Query
    rag = RAGService()
    request = ChatRequest(
        question="What is this document about?",
        llm="openai",
        doc_ids=[doc_id],
        stream=False,
    )
    result = await rag.query(request)

    assert result["answer"] != ""
    assert result["query_type"] in (
        "factual", "analytical", "general", "multi_hop", "tabular"
    )

    # Cleanup
    await ingestion.delete_document(doc_id)


@pytest.mark.integration
async def test_chat_greeting():
    """Greeting should not trigger retrieval."""
    from backend.app.services.rag import RAGService

    rag = RAGService()
    request = ChatRequest(
        question="Hello!",
        llm="openai",
        stream=False,
    )
    result = await rag.query(request)

    assert result["answer"] != ""
    assert result["query_type"] == "greeting"
    assert result["sources"] == []


@pytest.mark.integration
async def test_chat_streaming():
    """Streaming should yield META + tokens + DONE."""
    from backend.app.services.rag import RAGService

    rag = RAGService()
    request = ChatRequest(
        question="Hello!",
        llm="openai",
    )

    events = []
    async for event in rag.stream_query(request):
        events.append(event)

    assert any("__META__" in e for e in events)
    assert events[-1] == "[DONE]"
