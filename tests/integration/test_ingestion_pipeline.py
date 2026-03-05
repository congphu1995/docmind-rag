"""
Requires: Qdrant + PostgreSQL running (docker compose up qdrant postgres)
Run with: pytest tests/integration/ -m integration
"""
import os

import pytest


@pytest.mark.integration
async def test_full_ingestion_pipeline(sample_pdf_path):
    from backend.app.services.ingestion import IngestionService

    service = IngestionService()
    result = await service.ingest(
        file_path=sample_pdf_path,
        doc_name="sample.pdf",
        language="en",
    )

    assert result["doc_id"] is not None
    assert result["child_chunks"] > 0
    assert result["parent_chunks"] > 0
    assert result["child_chunks"] >= result["parent_chunks"]


@pytest.mark.integration
async def test_delete_removes_from_stores(sample_pdf_path):
    from backend.app.services.ingestion import IngestionService
    from backend.app.vectorstore.qdrant_client import QdrantWrapper

    service = IngestionService()
    result = await service.ingest(sample_pdf_path, "test.pdf")
    doc_id = result["doc_id"]

    await service.delete_document(doc_id)

    qdrant = QdrantWrapper()
    search_results = await qdrant.search(
        vector=[0.0] * 1536,
        filters={"doc_ids": [doc_id]},
    )
    assert len(search_results) == 0
