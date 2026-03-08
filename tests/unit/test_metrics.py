# tests/unit/test_metrics.py
from backend.app.core.metrics import (
    LLM_REQUEST_DURATION,
    LLM_TOKENS_TOTAL,
    HTTP_REQUEST_DURATION,
    INGESTION_STAGE_DURATION,
    CHUNKS_CREATED_TOTAL,
)


def test_metrics_are_defined():
    assert LLM_REQUEST_DURATION is not None
    assert LLM_TOKENS_TOTAL is not None
    assert HTTP_REQUEST_DURATION is not None
    assert INGESTION_STAGE_DURATION is not None
    assert CHUNKS_CREATED_TOTAL is not None
