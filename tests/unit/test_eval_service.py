import json
import pytest
from unittest.mock import patch
from pathlib import Path


def test_calculate_hit_rate():
    """Verify hit rate calculation."""
    from eval.run_eval import calculate_hit_rate

    results = [
        {"relevant_found": True},
        {"relevant_found": True},
        {"relevant_found": False},
        {"relevant_found": True},
    ]
    assert calculate_hit_rate(results) == 0.75


def test_calculate_hit_rate_empty():
    from eval.run_eval import calculate_hit_rate

    assert calculate_hit_rate([]) == 0.0


def test_load_manifest_missing():
    """Verify clear error when manifest missing."""
    from eval.run_eval import load_manifest

    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="make seed"):
            load_manifest()


def test_load_manifest_parses_doc_ids():
    """Verify manifest parsing extracts doc_ids."""
    from eval.run_eval import load_manifest

    manifest = {
        "user_id": "test-123",
        "documents": [
            {"doc_name": "AAPL.pdf", "doc_id": "d1", "hf_doc_name": "APPLE"},
            {"doc_name": "MSFT.pdf", "doc_id": "d2", "hf_doc_name": "MICROSOFT"},
        ],
    }
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "read_text", return_value=json.dumps(manifest)):
            result = load_manifest()
            assert result["user_id"] == "test-123"
            assert result["doc_ids"] == ["d1", "d2"]


def test_load_custom_dataset():
    """Verify custom dataset JSON parsing."""
    from scripts.seed_custom_eval import load_custom_dataset

    dataset = {
        "dataset": "docmind-custom",
        "version": "1.0",
        "papers": [
            {
                "paper_id": "attention-is-all-you-need",
                "title": "Attention Is All You Need",
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
                "filename": "attention_is_all_you_need.pdf",
            }
        ],
        "questions": [],
    }
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "read_text", return_value=json.dumps(dataset)):
            result = load_custom_dataset()
            assert result["version"] == "1.0"
            assert len(result["papers"]) == 1
            assert result["papers"][0]["paper_id"] == "attention-is-all-you-need"


def test_load_custom_dataset_missing():
    """Verify clear error when dataset file missing."""
    from scripts.seed_custom_eval import load_custom_dataset

    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="custom_dataset.json"):
            load_custom_dataset()
