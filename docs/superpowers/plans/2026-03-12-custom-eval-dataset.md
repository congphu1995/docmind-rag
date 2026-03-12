# Custom Evaluation Dataset Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a custom eval dataset of ML/AI papers with hand-verified Q&A pairs, alongside the existing FinanceBench eval.

**Architecture:** New dataset JSON file + seed script mirrors existing FinanceBench pattern. `run_eval.py` gains a `--dataset` flag to switch between financebench/custom. Custom eval scopes questions to individual papers (per-paper doc_ids) and adds per-type metric breakdowns.

**Tech Stack:** Python, httpx, asyncio, RAGAS, argparse. Same deps as existing eval.

---

## Chunk 1: Dataset File, Seed Script, and Tests

### Task 1: Create the custom dataset JSON

**Files:**
- Create: `eval/datasets/custom_dataset.json`

This file contains the paper list and starter Q&A pairs (3 examples for "Attention Is All You Need"). More questions will be added manually later per the spec's Q&A generation workflow.

- [ ] **Step 1: Create the dataset file**

```json
{
  "dataset": "docmind-custom",
  "version": "1.0",
  "papers": [
    {
      "paper_id": "attention-is-all-you-need",
      "title": "Attention Is All You Need",
      "pdf_url": "https://arxiv.org/pdf/1706.03762",
      "filename": "attention_is_all_you_need.pdf"
    },
    {
      "paper_id": "bert",
      "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      "pdf_url": "https://arxiv.org/pdf/1810.04805",
      "filename": "bert.pdf"
    },
    {
      "paper_id": "resnet",
      "title": "Deep Residual Learning for Image Recognition",
      "pdf_url": "https://arxiv.org/pdf/1512.03385",
      "filename": "resnet.pdf"
    },
    {
      "paper_id": "gpt2",
      "title": "Language Models are Unsupervised Multitask Learners",
      "pdf_url": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
      "filename": "gpt2.pdf"
    },
    {
      "paper_id": "adam",
      "title": "Adam: A Method for Stochastic Optimization",
      "pdf_url": "https://arxiv.org/pdf/1412.6980",
      "filename": "adam.pdf"
    },
    {
      "paper_id": "batch-norm",
      "title": "Batch Normalization: Accelerating Deep Network Training",
      "pdf_url": "https://arxiv.org/pdf/1502.03167",
      "filename": "batch_norm.pdf"
    },
    {
      "paper_id": "yolo",
      "title": "You Only Look Once: Unified, Real-Time Object Detection",
      "pdf_url": "https://arxiv.org/pdf/1506.02640",
      "filename": "yolo.pdf"
    },
    {
      "paper_id": "gan",
      "title": "Generative Adversarial Nets",
      "pdf_url": "https://arxiv.org/pdf/1406.2661",
      "filename": "gan.pdf"
    },
    {
      "paper_id": "dropout",
      "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
      "pdf_url": "https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf",
      "filename": "dropout.pdf"
    },
    {
      "paper_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "pdf_url": "https://arxiv.org/pdf/2106.09685",
      "filename": "lora.pdf"
    }
  ],
  "questions": [
    {
      "id": "att-f1",
      "paper_id": "attention-is-all-you-need",
      "question": "What was the BLEU score of the big Transformer model on the EN-DE WMT 2014 translation task?",
      "answer": "28.4",
      "type": "factual",
      "source_section": "Table 2",
      "source_page": [8],
      "difficulty": "easy"
    },
    {
      "id": "att-t1",
      "paper_id": "attention-is-all-you-need",
      "question": "How many parameters does the Transformer (big) model have compared to the base model?",
      "answer": "The big model has 213M parameters compared to 65M for the base model",
      "type": "table_reasoning",
      "source_section": "Table 3",
      "source_page": [9],
      "difficulty": "medium"
    },
    {
      "id": "att-m1",
      "paper_id": "attention-is-all-you-need",
      "question": "Why is self-attention faster than recurrent layers for typical NMT sequence lengths?",
      "answer": "Self-attention has O(n^2*d) complexity while recurrent layers have O(n*d^2). Since sequence length n is typically smaller than representation dimensionality d in NMT, self-attention is faster.",
      "type": "multi_hop",
      "source_section": "Table 1 + Section 4",
      "source_page": [5, 6],
      "difficulty": "hard"
    }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add eval/datasets/custom_dataset.json
git commit -m "feat: add custom eval dataset with 10 ML papers and starter questions"
```

---

### Task 2: Create the seed script with tests

**Files:**
- Create: `scripts/seed_custom_eval.py`
- Modify: `tests/unit/test_eval_service.py`

The seed script mirrors `scripts/seed_demo_data.py` but reads papers from `custom_dataset.json` and downloads from arXiv PDF URLs.

- [ ] **Step 1: Write tests for seed script helpers**

Add to `tests/unit/test_eval_service.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_service.py::test_load_custom_dataset tests/unit/test_eval_service.py::test_load_custom_dataset_missing -v`
Expected: FAIL — `scripts.seed_custom_eval` does not exist yet.

- [ ] **Step 3: Write the seed script**

Create `scripts/seed_custom_eval.py`:

```python
#!/usr/bin/env python3
"""
Seed custom eval documents (ML/AI papers from arXiv).
Downloads PDFs and ingests them via IngestionService.
Usage: uv run python scripts/seed_custom_eval.py
Requires: Docker services running (qdrant, postgres, redis)
"""

import asyncio
import json
import secrets
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EVAL_USER_EMAIL = "eval@docmind.system"
EVAL_USER_NAME = "eval_system"
DATASET_PATH = Path("eval/datasets/custom_dataset.json")
MANIFEST_PATH = Path("eval/datasets/custom_manifest.json")


def load_custom_dataset() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Create the dataset file first."
        )
    return json.loads(DATASET_PATH.read_text())


async def get_or_create_eval_user() -> str:
    from backend.app.core.database import AsyncSessionLocal, create_tables
    from backend.app.models.user import User
    from sqlalchemy import select

    await create_tables()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.email == EVAL_USER_EMAIL)
        )
        user = result.scalar_one_or_none()
        if user:
            print(f"Eval user exists: {user.id}")
            return user.id

    from backend.app.services.auth import AuthService

    auth = AuthService()
    user = await auth.register(
        email=EVAL_USER_EMAIL,
        username=EVAL_USER_NAME,
        password=secrets.token_urlsafe(32),
    )
    print(f"Created eval user: {user.id}")
    return user.id


async def download_papers(papers: list[dict], tmp_dir: Path) -> list[dict]:
    import httpx

    docs = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for paper in papers:
            pdf_path = tmp_dir / paper["filename"]
            print(f"  Downloading {paper['title'][:60]}...")
            r = await client.get(paper["pdf_url"])
            if r.status_code != 200 or len(r.content) < 1000:
                raise RuntimeError(
                    f"Download failed for {paper['paper_id']}: "
                    f"status={r.status_code}, size={len(r.content)}"
                )
            pdf_path.write_bytes(r.content)
            docs.append(
                {
                    "paper_id": paper["paper_id"],
                    "filename": paper["filename"],
                    "path": str(pdf_path),
                }
            )
            print(f"    OK ({len(r.content) // 1024} KB)")

    return docs


async def ingest_docs(docs: list[dict], user_id: str) -> list[dict]:
    from backend.app.services.ingestion import IngestionService

    service = IngestionService()

    manifest_docs = []
    for doc in docs:
        print(f"Ingesting: {doc['filename']}")
        try:
            result = await service.ingest(
                file_path=doc["path"],
                doc_name=doc["filename"],
                user_id=user_id,
            )
            manifest_docs.append(
                {
                    "paper_id": doc["paper_id"],
                    "filename": doc["filename"],
                    "doc_id": result["doc_id"],
                    "elements_parsed": result["elements_parsed"],
                    "child_chunks": result["child_chunks"],
                }
            )
            print(
                f"  Done: {result['elements_parsed']} elements, "
                f"{result['child_chunks']} chunks indexed"
            )
        except Exception as e:
            # Intentional deviation from seed_demo_data.py: fail fast on ingestion
            # errors because every paper must succeed for a curated eval dataset.
            print(f"  Failed: {e}")
            raise

    return manifest_docs


async def save_manifest(user_id: str, docs: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "user_id": user_id,
        "documents": docs,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved: {MANIFEST_PATH}")


async def main():
    print("=== DocMind RAG — Seed Custom Eval Papers ===\n")

    dataset = load_custom_dataset()
    papers = dataset["papers"]
    print(f"Dataset v{dataset['version']}: {len(papers)} papers\n")

    user_id = await get_or_create_eval_user()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("\nDownloading papers...")
        docs = await download_papers(papers, tmp_dir)
        print(f"Ready to ingest: {len(docs)} papers\n")

        ingested = await ingest_docs(docs, user_id)

    await save_manifest(user_id, ingested)
    print(f"\nSeeding complete. {len(ingested)} papers ingested.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_service.py::test_load_custom_dataset tests/unit/test_eval_service.py::test_load_custom_dataset_missing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/seed_custom_eval.py tests/unit/test_eval_service.py
git commit -m "feat: add custom eval seed script for ML/AI papers"
```

---

## Chunk 2: Eval Runner Extension and Makefile

### Task 3: Extend `run_eval.py` with `--dataset` flag and per-type metrics

**Files:**
- Modify: `eval/run_eval.py`
- Modify: `tests/unit/test_eval_service.py`

This is the core change. The eval runner needs to:
1. Accept `--dataset financebench|custom` via argparse
2. Load questions from local JSON for custom dataset
3. Build `paper_id → doc_id` lookup and scope per question
4. Compute per-type metric breakdowns

- [ ] **Step 1: Write tests for custom question loading and per-type metrics**

Add to `tests/unit/test_eval_service.py`:

```python
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
    with patch.object(Path, "read_text", return_value=json.dumps(dataset)):
        questions = load_custom_questions(Path("fake.json"), sample_size=10)
        assert len(questions) == 2
        assert questions[0]["question"] == "What was the BLEU score?"
        assert questions[0]["paper_id"] == "attention-is-all-you-need"
        assert questions[0]["type"] == "factual"

        # Verify sample_size truncation
        questions_limited = load_custom_questions(Path("fake.json"), sample_size=1)
        assert len(questions_limited) == 1


def test_load_custom_manifest():
    """Verify custom manifest builds paper_id → doc_id lookup."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_service.py::test_load_custom_questions tests/unit/test_eval_service.py::test_load_custom_manifest tests/unit/test_eval_service.py::test_load_custom_manifest_missing tests/unit/test_eval_service.py::test_compute_metrics_by_type -v`
Expected: FAIL — functions don't exist yet.

- [ ] **Step 3: Implement changes to `eval/run_eval.py`**

**Important:** Preserve all existing code unchanged: imports, `load_dotenv()`, existing constants (`MANIFEST_PATH`, `RESULTS_PATH`, `SAMPLE_SIZE`), and all existing functions (`load_manifest`, `load_questions`, `calculate_hit_rate`, `compute_ragas_metrics`). Only `run_eval()` and `main()` are replaced. All new code below is added alongside the existing code.

Add these constants near the top (after existing constants):

```python
CUSTOM_DATASET_PATH = Path("eval/datasets/custom_dataset.json")
CUSTOM_MANIFEST_PATH = Path("eval/datasets/custom_manifest.json")
CUSTOM_RESULTS_PATH = Path("eval/results/custom_results.json")
```

Add `argparse` to the imports:

```python
import argparse
```

Add these new functions after the existing `load_manifest()`:

```python
def load_custom_manifest() -> dict:
    if not CUSTOM_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Custom manifest not found. Run `make seed-custom` first."
        )
    data = json.loads(CUSTOM_MANIFEST_PATH.read_text())
    paper_to_doc = {d["paper_id"]: d["doc_id"] for d in data["documents"]}
    return {
        "user_id": data["user_id"],
        "paper_to_doc": paper_to_doc,
        "documents": data["documents"],
    }


def load_custom_questions(dataset_path: Path, sample_size: int) -> list[dict]:
    data = json.loads(dataset_path.read_text())
    questions = data["questions"][:sample_size]
    return [
        {
            "question": q["question"],
            "answer": q["answer"],
            "paper_id": q["paper_id"],
            "type": q["type"],
            "difficulty": q.get("difficulty", ""),
        }
        for q in questions
    ]


def compute_metrics_by_type(results: list[dict]) -> dict:
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        qtype = r.get("type", "unknown")
        by_type[qtype].append(r)

    output = {}
    for qtype, items in by_type.items():
        hits = sum(1 for r in items if r.get("relevant_found"))
        output[qtype] = {
            "retrieval_hit_rate": round(hits / len(items), 4) if items else 0.0,
            "sample_size": len(items),
        }
    return output
```

Modify `run_eval()` to accept an optional `paper_to_doc` dict. When present, use per-paper scoping. Also propagate question `type` into results:

Replace the existing `run_eval` function with:

```python
async def run_eval(
    questions: list[dict],
    doc_ids: list[str],
    paper_to_doc: dict[str, str] | None = None,
) -> dict:
    from backend.app.services.rag import RAGService
    from backend.app.schemas.chat import ChatRequest

    service = RAGService()
    results = []
    latencies = []

    for i, q in enumerate(questions):
        start = time.time()

        # Per-paper scoping for custom dataset
        if paper_to_doc and q.get("paper_id"):
            paper_id = q["paper_id"]
            if paper_id not in paper_to_doc:
                raise ValueError(
                    f"Question references paper_id '{paper_id}' not found in manifest. "
                    f"Run `make seed-custom` to ingest all papers first."
                )
            q_doc_ids = [paper_to_doc[paper_id]]
        else:
            q_doc_ids = doc_ids

        request = ChatRequest(
            question=q["question"],
            llm="openai",
            doc_ids=q_doc_ids,
            stream=False,
        )

        try:
            response = await service.query(request)
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

            reranked = response.get("reranked_chunks", [])
            contexts = [
                c.get("content", "") for c in reranked if c.get("content")
            ]

            results.append(
                {
                    "question": q["question"],
                    "ground_truth": q.get("answer", ""),
                    "generated_answer": response.get("answer", ""),
                    "contexts": contexts,
                    "relevant_found": any(
                        c.get("score", 0) > 0.5 for c in reranked
                    ),
                    "query_type": response.get("query_type", ""),
                    "hyde_used": response.get("hyde_used", False),
                    "type": q.get("type", ""),
                }
            )

            status = "OK" if results[-1]["relevant_found"] else "MISS"
            print(
                f"  [{i + 1}/{len(questions)}] {status} ({elapsed_ms:.0f}ms) {q['question'][:60]}..."
            )
        except Exception as e:
            print(f"  [{i + 1}/{len(questions)}] ERROR: {e}")
            results.append(
                {
                    "question": q["question"],
                    "ground_truth": q.get("answer", ""),
                    "generated_answer": "",
                    "contexts": [],
                    "relevant_found": False,
                    "type": q.get("type", ""),
                    "error": str(e),
                }
            )

    return {"results": results, "latencies": latencies}
```

Replace the existing `main()` with a version that handles both datasets:

```python
async def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--dataset",
        choices=["financebench", "custom"],
        default="financebench",
        help="Dataset to evaluate against (default: financebench)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Max questions to evaluate",
    )
    args = parser.parse_args()

    if args.dataset == "custom":
        await run_custom_eval(args.sample_size or 100)
    else:
        await run_financebench_eval(args.sample_size or SAMPLE_SIZE)


async def run_financebench_eval(sample_size: int):
    print("=== DocMind RAG — FinanceBench Evaluation ===\n")

    manifest = load_manifest()
    print(f"Eval docs: {len(manifest['doc_ids'])} documents")

    questions = await load_questions(manifest, sample_size)
    print(f"Questions: {len(questions)} matched to seeded docs\n")

    print("Running RAG pipeline...")
    eval_data = await run_eval(questions, manifest["doc_ids"])
    results = eval_data["results"]
    latencies = eval_data["latencies"]

    hit_rate = calculate_hit_rate(results)
    ragas = await compute_ragas_metrics(results)

    latencies.sort()
    p95_idx = int(len(latencies) * 0.95)
    latency_p95 = round(latencies[p95_idx] if latencies else 0, 1)

    metrics = {
        "retrieval_hit_rate": hit_rate,
        **ragas,
        "latency_p95_ms": latency_p95,
        "sample_size": len(results),
    }

    print_metrics("FinanceBench Evaluation", metrics)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": "financebench",
        "config": "default (parent-child 800/150, enrichment, tables atomic)",
        "metrics": metrics,
        "per_question": results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


async def run_custom_eval(sample_size: int):
    print("=== DocMind RAG — Custom Dataset Evaluation ===\n")

    manifest = load_custom_manifest()
    paper_to_doc = manifest["paper_to_doc"]
    print(f"Eval docs: {len(paper_to_doc)} papers")

    dataset = json.loads(CUSTOM_DATASET_PATH.read_text())
    version = dataset.get("version", "unknown")

    questions = load_custom_questions(CUSTOM_DATASET_PATH, sample_size)
    print(f"Questions: {len(questions)} (dataset v{version})\n")

    print("Running RAG pipeline...")
    all_doc_ids = list(paper_to_doc.values())
    eval_data = await run_eval(questions, all_doc_ids, paper_to_doc=paper_to_doc)
    results = eval_data["results"]
    latencies = eval_data["latencies"]

    hit_rate = calculate_hit_rate(results)
    ragas = await compute_ragas_metrics(results)

    latencies.sort()
    p95_idx = int(len(latencies) * 0.95)
    latency_p95 = round(latencies[p95_idx] if latencies else 0, 1)

    overall_metrics = {
        "retrieval_hit_rate": hit_rate,
        **ragas,
        "latency_p95_ms": latency_p95,
        "sample_size": len(results),
    }

    # Per-type breakdown: hit rate from results, RAGAS per type
    by_type_hit = compute_metrics_by_type(results)

    # Group results by type for per-type RAGAS
    from collections import defaultdict

    results_by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        results_by_type[r.get("type", "unknown")].append(r)

    by_type = {}
    for qtype, type_results in results_by_type.items():
        type_ragas = await compute_ragas_metrics(type_results)
        by_type[qtype] = {
            **by_type_hit.get(qtype, {}),
            **type_ragas,
        }

    print_metrics("Custom Dataset Evaluation", overall_metrics)
    print("\nPer-type breakdown:")
    for qtype, type_metrics in by_type.items():
        print(f"  {qtype}: hit_rate={type_metrics.get('retrieval_hit_rate', 0):.1%}, "
              f"faithfulness={type_metrics.get('faithfulness', 0):.4f}, "
              f"relevancy={type_metrics.get('answer_relevancy', 0):.4f}, "
              f"recall={type_metrics.get('context_recall', 0):.4f}")

    CUSTOM_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": "docmind-custom",
        "version": version,
        "config": "default (parent-child 800/150, enrichment, tables atomic)",
        "metrics": {
            "overall": overall_metrics,
            "by_type": by_type,
        },
        "per_question": results,
    }
    CUSTOM_RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {CUSTOM_RESULTS_PATH}")


def print_metrics(title: str, metrics: dict):
    print(f"\n{'=' * 50}")
    print(f"  DocMind RAG — {title}")
    print(f"{'=' * 50}")
    print(f"  Retrieval Hit Rate:  {metrics['retrieval_hit_rate']:.1%}")
    print(f"  Faithfulness:        {metrics['faithfulness']:.4f}")
    print(f"  Answer Relevancy:    {metrics['answer_relevancy']:.4f}")
    print(f"  Context Recall:      {metrics['context_recall']:.4f}")
    print(f"  Latency p95:         {metrics['latency_p95_ms']:.0f}ms")
    print(f"  Sample Size:         {metrics['sample_size']}")
    print(f"{'=' * 50}")
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_service.py -v`
Expected: All 10 tests PASS (4 existing + 2 from Task 2 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add eval/run_eval.py tests/unit/test_eval_service.py
git commit -m "feat: extend eval runner with --dataset flag and per-type metrics"
```

---

### Task 4: Add Makefile targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add seed-custom and eval-custom targets**

Add after the existing `eval` and `seed` targets in `Makefile`:

```makefile
eval-custom:
	uv run python eval/run_eval.py --dataset custom

seed-custom:
	uv run python scripts/seed_custom_eval.py
```

- [ ] **Step 2: Update .PHONY**

Update the `.PHONY` line to include the new targets:

```makefile
.PHONY: dev test lint eval eval-custom seed seed-custom frontend backend infra clean
```

- [ ] **Step 3: Verify**

Run: `make -n eval-custom` and `make -n seed-custom`
Expected: Prints the commands without executing.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "feat: add seed-custom and eval-custom Makefile targets"
```

---

### Task 5: Run full test suite

- [ ] **Step 1: Run all unit tests**

Run: `uv run pytest tests/unit -v`
Expected: All tests PASS.

- [ ] **Step 2: Run linter**

Run: `uv run ruff check eval/ scripts/seed_custom_eval.py tests/unit/test_eval_service.py`
Expected: No errors.

- [ ] **Step 3: Fix any lint issues and commit if needed**

```bash
git add -A && git commit -m "fix: lint fixes"
```
