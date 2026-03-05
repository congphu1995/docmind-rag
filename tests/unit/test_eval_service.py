import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.services.eval import EvalService


async def test_eval_calculates_hit_rate():
    """Verify hit rate calculation from retrieval results."""
    service = EvalService.__new__(EvalService)
    results = [
        {"relevant_found": True},
        {"relevant_found": True},
        {"relevant_found": False},
        {"relevant_found": True},
    ]
    hit_rate = service._calculate_hit_rate(results)
    assert hit_rate == 0.75


async def test_eval_run_creates_record():
    """Verify eval run creates a DB record and returns run_id."""
    with patch("backend.app.services.eval.AsyncSessionLocal") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        service = EvalService.__new__(EvalService)
        run_id = await service._create_run_record(
            dataset="financebench", sample_size=10, config={}
        )
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
