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


def test_load_custom_questions():
    """Verify loading questions from custom dataset JSON."""
    from eval.run_eval import load_custom_questions

    dataset = {
        "dataset": "docmind-custom",
        "version": "1.0",
        "papers": [],
        "questions": [
            {
                "id": "att-f1",
                "paper_id": "attention-is-all-you-need",
                "question": "What was the BLEU score?",
                "answer": "28.4",
                "type": "factual",
                "source_section": "Table 2",
                "source_page": [8],
                "difficulty": "easy",
            },
            {
                "id": "att-m1",
                "paper_id": "attention-is-all-you-need",
                "question": "Why is self-attention faster?",
                "answer": "Because n < d typically",
                "type": "multi_hop",
                "source_section": "Table 1 + Section 4",
                "source_page": [5, 6],
                "difficulty": "hard",
            },
        ],
    }
    questions = load_custom_questions(dataset, sample_size=10)
    assert len(questions) == 2
    assert questions[0]["question"] == "What was the BLEU score?"
    assert questions[0]["paper_id"] == "attention-is-all-you-need"
    assert questions[0]["type"] == "factual"

    # Verify sample_size truncation
    questions_limited = load_custom_questions(dataset, sample_size=1)
    assert len(questions_limited) == 1


def test_load_custom_manifest():
    """Verify custom manifest builds paper_id -> doc_id lookup."""
    from eval.run_eval import load_custom_manifest

    manifest = {
        "user_id": "user-123",
        "documents": [
            {"paper_id": "attention-is-all-you-need", "doc_id": "d1", "filename": "att.pdf"},
            {"paper_id": "bert", "doc_id": "d2", "filename": "bert.pdf"},
        ],
    }
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "read_text", return_value=json.dumps(manifest)):
            result = load_custom_manifest()
            assert result["user_id"] == "user-123"
            assert result["paper_to_doc"]["attention-is-all-you-need"] == "d1"
            assert result["paper_to_doc"]["bert"] == "d2"
            assert len(result["documents"]) == 2


def test_load_custom_manifest_missing():
    """Verify clear error when custom manifest is missing."""
    from eval.run_eval import load_custom_manifest

    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="seed-custom"):
            load_custom_manifest()


def test_compute_metrics_by_type():
    """Verify per-type metric aggregation."""
    from eval.run_eval import compute_metrics_by_type

    results = [
        {"type": "factual", "relevant_found": True},
        {"type": "factual", "relevant_found": True},
        {"type": "factual", "relevant_found": False},
        {"type": "table_reasoning", "relevant_found": True},
        {"type": "multi_hop", "relevant_found": False},
    ]
    by_type = compute_metrics_by_type(results)
    assert set(by_type.keys()) == {"factual", "table_reasoning", "multi_hop"}
    # factual: 2/3 hits
    assert by_type["factual"]["retrieval_hit_rate"] == round(2 / 3, 4)
    assert by_type["factual"]["sample_size"] == 3
    # table_reasoning: 1/1
    assert by_type["table_reasoning"]["retrieval_hit_rate"] == 1.0
    # multi_hop: 0/1
    assert by_type["multi_hop"]["retrieval_hit_rate"] == 0.0
