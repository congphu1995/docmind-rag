# Eval Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the RAG evaluation pipeline work end-to-end: seed FinanceBench docs, fix RAGAS v0.4 API, run eval via script, produce committed result JSON, notebook for visualization only.

**Architecture:** Pre-seed 5 FinanceBench 10-K PDFs under a system "eval" user. `eval/run_eval.py` script calls RAGService directly (no HTTP, no Celery), computes RAGAS metrics, saves JSON. Notebook loads the JSON and renders formatted results for GitHub display. Remove eval API endpoint and Celery task — they add complexity with no value.

**Tech Stack:** RAGAS v0.4.3, langchain-openai (for RAGAS LLM/embeddings), HuggingFace datasets

> **Depends on:** Ingestion pipeline, RAGService, auth system — all already built.

---

## Task 1: Add langchain-openai Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add the dependency**

```bash
uv add langchain-openai
```

RAGAS v0.4.3 requires `langchain_openai.ChatOpenAI` and `langchain_openai.OpenAIEmbeddings` to pass as `llm` and `embeddings` params to metric constructors.

**Step 2: Verify import**

Run: `uv run python -c "from langchain_openai import ChatOpenAI, OpenAIEmbeddings; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add langchain-openai for RAGAS eval metrics"
```

---

## Task 2: Seed Script — Create Eval User + Ingest FinanceBench PDFs

**Files:**
- Rewrite: `scripts/seed_demo_data.py`

**Step 1: Write the seed script**

Rewrite `scripts/seed_demo_data.py`:

```python
#!/usr/bin/env python3
"""
Seed FinanceBench documents for evaluation.
Creates a system eval user, downloads 5 10-K PDFs, ingests them.
Usage: uv run python scripts/seed_demo_data.py
Requires: Docker services running (qdrant, postgres, redis)
"""
import asyncio
import json
import secrets
import sys
import tempfile
from pathlib import Path

EVAL_USER_EMAIL = "eval@docmind.system"
EVAL_USER_NAME = "eval_system"
MANIFEST_PATH = Path("eval/datasets/seed_manifest.json")

# 5 FinanceBench companies — deterministic set
# These are referenced in the FinanceBench dataset questions
FINANCEBENCH_DOCS = [
    {
        "doc_name": "AAPL_10K_2023.pdf",
        "hf_doc_name": "APPLE INC",
    },
    {
        "doc_name": "MSFT_10K_2023.pdf",
        "hf_doc_name": "MICROSOFT CORP",
    },
    {
        "doc_name": "AMZN_10K_2023.pdf",
        "hf_doc_name": "AMAZON COM INC",
    },
    {
        "doc_name": "GOOG_10K_2023.pdf",
        "hf_doc_name": "ALPHABET INC",
    },
    {
        "doc_name": "META_10K_2023.pdf",
        "hf_doc_name": "META PLATFORMS INC",
    },
]


async def get_or_create_eval_user() -> str:
    """Create eval system user if not exists, return user_id."""
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

    # Create new eval user
    from backend.app.services.auth import AuthService
    auth = AuthService()
    user = await auth.register(
        email=EVAL_USER_EMAIL,
        username=EVAL_USER_NAME,
        password=secrets.token_urlsafe(32),
    )
    print(f"Created eval user: {user.id}")
    return user.id


async def download_financebench_pdfs(tmp_dir: Path) -> list[dict]:
    """
    Download FinanceBench dataset and extract PDF links.
    Falls back to creating synthetic text files if PDFs unavailable.
    """
    docs = []
    try:
        from datasets import load_dataset

        ds = load_dataset("PatronusAI/financebench", split="train")

        # Get unique doc_link entries for our target companies
        seen = set()
        for item in ds:
            doc_name = item.get("doc_name", "")
            doc_link = item.get("doc_link", "")
            if doc_link and doc_name not in seen:
                for target in FINANCEBENCH_DOCS:
                    if target["hf_doc_name"].lower() in doc_name.lower():
                        seen.add(doc_name)
                        # Download PDF
                        pdf_path = tmp_dir / target["doc_name"]
                        if not pdf_path.exists():
                            import httpx
                            print(f"  Downloading {target['doc_name']}...")
                            try:
                                async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                                    r = await client.get(doc_link)
                                    if r.status_code == 200 and len(r.content) > 1000:
                                        pdf_path.write_bytes(r.content)
                                        docs.append({
                                            "doc_name": target["doc_name"],
                                            "hf_doc_name": doc_name,
                                            "path": str(pdf_path),
                                        })
                                    else:
                                        print(f"    Skip: status={r.status_code}, size={len(r.content)}")
                            except Exception as e:
                                print(f"    Download failed: {e}")
                        break
            if len(docs) >= 5:
                break
    except Exception as e:
        print(f"FinanceBench download failed: {e}")

    if not docs:
        print("Creating synthetic seed documents as fallback...")
        docs = create_synthetic_docs(tmp_dir)

    return docs


def create_synthetic_docs(tmp_dir: Path) -> list[dict]:
    """Create simple text files as fallback seed data."""
    docs = []
    for i, company in enumerate(["Acme Corp", "Beta Inc", "Gamma LLC", "Delta Co", "Echo Ltd"]):
        path = tmp_dir / f"synthetic_{i+1}.txt"
        path.write_text(
            f"{company} Annual Report 2023\n\n"
            f"Financial Overview\n"
            f"Total revenue for fiscal year 2023 was ${(i+1)*10}B.\n"
            f"Operating income was ${(i+1)*2}B.\n\n"
            f"Business Segments\n"
            f"{company} operates in {i+2} segments globally.\n"
            f"The largest segment contributed {50+i*5}% of revenue.\n\n"
            f"Risk Factors\n"
            f"Key risks include market competition and regulatory changes.\n"
        )
        docs.append({"doc_name": path.name, "path": str(path), "hf_doc_name": company})
    return docs


async def ingest_docs(docs: list[dict], user_id: str) -> list[dict]:
    """Ingest documents via IngestionService, return doc_id mapping."""
    from backend.app.services.ingestion import IngestionService
    service = IngestionService()

    manifest_docs = []
    for doc in docs:
        print(f"Ingesting: {doc['doc_name']}")
        try:
            result = await service.ingest(
                file_path=doc["path"],
                doc_name=doc["doc_name"],
                user_id=user_id,
            )
            manifest_docs.append({
                "doc_name": doc["doc_name"],
                "hf_doc_name": doc["hf_doc_name"],
                "doc_id": result["doc_id"],
                "elements_parsed": result["elements_parsed"],
                "child_chunks": result["child_chunks"],
            })
            print(
                f"  Done: {result['elements_parsed']} elements, "
                f"{result['child_chunks']} chunks indexed"
            )
        except Exception as e:
            print(f"  Failed: {e}")

    return manifest_docs


async def save_manifest(user_id: str, docs: list[dict]):
    """Save seed manifest for eval to reference."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "user_id": user_id,
        "documents": docs,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved: {MANIFEST_PATH}")


async def main():
    print("=== DocMind RAG — Seed FinanceBench ===\n")

    user_id = await get_or_create_eval_user()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("\nDownloading FinanceBench documents...")
        docs = await download_financebench_pdfs(tmp_dir)
        print(f"Ready to ingest: {len(docs)} documents\n")

        ingested = await ingest_docs(docs, user_id)

    await save_manifest(user_id, ingested)
    print(f"\nSeeding complete. {len(ingested)} documents ingested.")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Verify syntax**

Run: `uv run python -c "from scripts.seed_demo_data import FINANCEBENCH_DOCS; print(len(FINANCEBENCH_DOCS))"`
Expected: `5`

**Step 3: Commit**

```bash
git add scripts/seed_demo_data.py
git commit -m "feat: seed script — eval user + FinanceBench PDF download + ingest"
```

---

## Task 3: Create Eval Runner Script

**Files:**
- Create: `eval/run_eval.py`

**Step 1: Write the eval script**

Create `eval/run_eval.py`:

```python
#!/usr/bin/env python3
"""
Run RAG evaluation against seeded FinanceBench documents.
Calls RAGService directly — no HTTP, no Celery.

Usage: uv run python eval/run_eval.py
Requires: `make seed` to have been run first.
"""
import asyncio
import json
import time
from pathlib import Path

MANIFEST_PATH = Path("eval/datasets/seed_manifest.json")
RESULTS_PATH = Path("eval/results/financebench_results.json")
SAMPLE_SIZE = 30


def load_manifest() -> dict:
    """Load seed manifest to get doc_ids."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Seed manifest not found. Run `make seed` first."
        )
    data = json.loads(MANIFEST_PATH.read_text())
    return {
        "user_id": data["user_id"],
        "doc_ids": [d["doc_id"] for d in data["documents"]],
        "documents": data["documents"],
    }


async def load_questions(sample_size: int) -> list[dict]:
    """Load FinanceBench questions matched to seeded companies."""
    from datasets import load_dataset

    # Load manifest to know which companies are seeded
    manifest = json.loads(MANIFEST_PATH.read_text())
    hf_names = {d["hf_doc_name"].lower() for d in manifest["documents"]}

    ds = load_dataset("PatronusAI/financebench", split="train")

    matched = []
    for item in ds:
        doc_name = item.get("doc_name", "").lower()
        if any(name in doc_name for name in hf_names):
            matched.append({
                "question": item["question"],
                "answer": item.get("answer", ""),
                "doc_name": item.get("doc_name", ""),
            })
        if len(matched) >= sample_size:
            break

    if not matched:
        print("WARNING: No matched questions found. Using first N from dataset.")
        for item in ds.select(range(min(sample_size, len(ds)))):
            matched.append({
                "question": item["question"],
                "answer": item.get("answer", ""),
                "doc_name": item.get("doc_name", ""),
            })

    return matched


async def run_eval(questions: list[dict], doc_ids: list[str]) -> dict:
    """Run RAG pipeline on each question, collect results."""
    from backend.app.services.rag import RAGService
    from backend.app.schemas.chat import ChatRequest

    service = RAGService()
    results = []
    latencies = []

    for i, q in enumerate(questions):
        start = time.time()
        request = ChatRequest(
            question=q["question"],
            llm="openai",
            doc_ids=doc_ids,
            stream=False,
        )

        try:
            response = await service.query(request)
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

            results.append({
                "question": q["question"],
                "ground_truth": q.get("answer", ""),
                "generated_answer": response.get("answer", ""),
                "contexts": [
                    s.get("content_preview", "")
                    for s in response.get("sources", [])
                ],
                "relevant_found": any(
                    s.get("score", 0) > 0.5
                    for s in response.get("sources", [])
                ),
                "query_type": response.get("query_type", ""),
                "hyde_used": response.get("hyde_used", False),
            })

            status = "OK" if results[-1]["relevant_found"] else "MISS"
            print(f"  [{i+1}/{len(questions)}] {status} ({elapsed_ms:.0f}ms) {q['question'][:60]}...")
        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] ERROR: {e}")
            results.append({
                "question": q["question"],
                "ground_truth": q.get("answer", ""),
                "generated_answer": "",
                "contexts": [],
                "relevant_found": False,
                "error": str(e),
            })

    return {"results": results, "latencies": latencies}


def calculate_hit_rate(results: list[dict]) -> float:
    """Percentage of questions where a relevant chunk was retrieved."""
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.get("relevant_found"))
    return round(hits / len(results), 4)


async def compute_ragas_metrics(results: list[dict]) -> dict:
    """Compute RAGAS metrics using v0.4.3 class-based API."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextRecall,
            Faithfulness,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["generated_answer"],
                reference=r["ground_truth"],
                retrieved_contexts=r["contexts"],
            )
            for r in results
            if r["generated_answer"] and r["contexts"]
        ]

        if not samples:
            print("WARNING: No valid samples for RAGAS evaluation.")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}

        print(f"\nComputing RAGAS metrics on {len(samples)} samples...")
        dataset = EvaluationDataset(samples=samples)
        metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextRecall(llm=llm),
        ]

        ragas_result = evaluate(dataset=dataset, metrics=metrics)

        return {
            "faithfulness": round(ragas_result["faithfulness"], 4),
            "answer_relevancy": round(ragas_result["answer_relevancy"], 4),
            "context_recall": round(ragas_result["context_recall"], 4),
        }
    except Exception as e:
        print(f"WARNING: RAGAS metrics failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}


async def main():
    print("=== DocMind RAG — Baseline Evaluation ===\n")

    # 1. Load manifest
    manifest = load_manifest()
    print(f"Eval docs: {len(manifest['doc_ids'])} documents")

    # 2. Load questions
    questions = await load_questions(SAMPLE_SIZE)
    print(f"Questions: {len(questions)} matched to seeded docs\n")

    # 3. Run RAG on each question
    print("Running RAG pipeline...")
    eval_data = await run_eval(questions, manifest["doc_ids"])
    results = eval_data["results"]
    latencies = eval_data["latencies"]

    # 4. Compute metrics
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

    # 5. Display results
    print(f"\n{'='*50}")
    print(f"  DocMind RAG — Baseline Evaluation Results")
    print(f"{'='*50}")
    print(f"  Retrieval Hit Rate:  {metrics['retrieval_hit_rate']:.1%}")
    print(f"  Faithfulness:        {metrics['faithfulness']:.4f}")
    print(f"  Answer Relevancy:    {metrics['answer_relevancy']:.4f}")
    print(f"  Context Recall:      {metrics['context_recall']:.4f}")
    print(f"  Latency p95:         {metrics['latency_p95_ms']:.0f}ms")
    print(f"  Sample Size:         {metrics['sample_size']}")
    print(f"{'='*50}")

    # 6. Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": "financebench",
        "config": "default (parent-child 800/150, enrichment, tables atomic)",
        "metrics": metrics,
        "per_question": results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Update Makefile**

In `Makefile`, replace the `eval` target:

```makefile
eval:
	uv run python eval/run_eval.py
```

**Step 3: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('eval/run_eval.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add eval/run_eval.py Makefile
git commit -m "feat: eval runner script — calls RAGService directly, RAGAS v0.4.3, saves JSON"
```

---

## Task 4: Remove Eval API Endpoint + Celery Task

**Files:**
- Delete: `backend/app/api/eval.py`
- Delete: `backend/app/workers/eval_tasks.py`
- Modify: `backend/app/main.py` — remove eval router registration
- Delete: `backend/app/services/eval.py` — no longer needed (logic moved to script)

**Step 1: Remove eval router from main.py**

In `backend/app/main.py`, remove:
- The import: `from backend.app.api import ... eval ...`
- The router: `app.include_router(eval.router, prefix="/api/v1/eval", tags=["eval"])`

**Step 2: Delete files**

```bash
rm backend/app/api/eval.py
rm backend/app/workers/eval_tasks.py
rm backend/app/services/eval.py
```

Keep `backend/app/models/eval.py` and `backend/app/schemas/eval.py` — they're still useful for storing eval run history if we add that later.

**Step 3: Verify backend still starts**

Run: `uv run python -c "from backend.app.main import app; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add -u
git commit -m "refactor: remove eval API + Celery task — replaced by direct script"
```

---

## Task 5: Update Baseline Notebook (Viz Only)

**Files:**
- Rewrite: `eval/notebooks/01_baseline_eval.ipynb`

**Step 1: Rewrite the notebook**

The notebook now just loads and displays saved results — no HTTP calls, no polling.

**Cell 0 (markdown):**
```markdown
# DocMind RAG — Baseline Evaluation Results

FinanceBench dataset (5 seeded 10-K filings).
Default config: Parent-Child (800/150), text-embedding-3-small, GPT-4o.

**To run eval:** `make seed && make eval`
This notebook displays the saved results.
```

**Cell 1 (code):**
```python
import json
from pathlib import Path

results_path = Path("../results/financebench_results.json")
if not results_path.exists():
    raise FileNotFoundError(
        "No results found. Run `make seed && make eval` first."
    )

data = json.loads(results_path.read_text())
metrics = data["metrics"]

print(f"Dataset:  {data['dataset']}")
print(f"Config:   {data['config']}")
print(f"Samples:  {metrics['sample_size']}")
```

**Cell 2 (code):**
```python
# Metrics summary
print(f"\n{'='*50}")
print(f"  Evaluation Metrics")
print(f"{'='*50}")
print(f"  Retrieval Hit Rate:  {metrics['retrieval_hit_rate']:.1%}")
print(f"  Faithfulness:        {metrics['faithfulness']:.4f}")
print(f"  Answer Relevancy:    {metrics['answer_relevancy']:.4f}")
print(f"  Context Recall:      {metrics['context_recall']:.4f}")
print(f"  Latency p95:         {metrics['latency_p95_ms']:.0f}ms")
print(f"{'='*50}")
```

**Cell 3 (code):**
```python
# Per-question breakdown
questions = data.get("per_question", [])
print(f"\n{'#':<4} {'Hit':<5} {'Type':<12} {'Question':<60}")
print("-" * 81)
for i, q in enumerate(questions, 1):
    hit = "Y" if q.get("relevant_found") else "N"
    qtype = q.get("query_type", "?")
    question = q["question"][:58]
    print(f"{i:<4} {hit:<5} {qtype:<12} {question}")
```

**Step 2: Remove ablation notebook** (skipping ablation)

```bash
rm eval/notebooks/02_chunking_ablation.ipynb
```

**Step 3: Commit**

```bash
git add eval/notebooks/01_baseline_eval.ipynb
git rm eval/notebooks/02_chunking_ablation.ipynb
git commit -m "feat: notebook shows saved results only, remove ablation notebook"
```

---

## Task 6: Update Eval README + Download Script

**Files:**
- Modify: `eval/README.md`
- Modify: `eval/datasets/download_financebench.py`

**Step 1: Update download script to save matched questions**

In `eval/datasets/download_financebench.py`, add after the main dump (after line 36):

```python
    # Also save questions matched to seed companies
    seed_companies = ["APPLE", "MICROSOFT", "AMAZON", "ALPHABET", "META PLATFORMS"]
    matched = [
        q for q in questions
        if any(c in q.get("doc_name", "").upper() for c in seed_companies)
    ]
    matched_path = output_dir / "financebench_matched.json"
    with open(matched_path, "w") as f:
        json.dump(matched, f, indent=2)
    print(f"Saved {len(matched)} matched questions (for 5 seed companies) to {matched_path}")
```

**Step 2: Rewrite eval README**

```markdown
# Evaluation

## Quick Start

```bash
# 1. Start services
docker compose up -d

# 2. Seed FinanceBench documents (downloads 5 10-K PDFs, ingests them)
make seed

# 3. Run evaluation
make eval

# 4. View results
# Open eval/notebooks/01_baseline_eval.ipynb
# Or: cat eval/results/financebench_results.json
```

## How It Works

```
make seed   → Download 5 10-K PDFs → Ingest via full pipeline → Save manifest
make eval   → Load questions → RAGService.query() per question → RAGAS metrics → Save JSON
```

The eval script calls `RAGService` directly (no HTTP/Celery). This tests the full pipeline:
- **Ingestion quality** — did chunking/embedding produce good vectors?
- **Retrieval quality** — does the agent find the right chunks?
- **Generation quality** — is the answer faithful to the context?

## Metrics

| Metric | Tool | Target |
|---|---|---|
| Retrieval Hit Rate | Custom | > 80% |
| Faithfulness | RAGAS | > 0.85 |
| Answer Relevancy | RAGAS | > 0.80 |
| Context Recall | RAGAS | > 0.75 |
| Latency p95 | Logged | < 3s |

## Reproduce

1. `make seed` — ingest eval documents
2. `make eval` — run evaluation
3. Results: `eval/results/financebench_results.json`
4. Notebook: `eval/notebooks/01_baseline_eval.ipynb`
```

**Step 3: Commit**

```bash
git add eval/datasets/download_financebench.py eval/README.md
git commit -m "docs: update eval README for script-based flow, add matched questions to downloader"
```

---

## Task 7: Clean Up Tests + Final Verification

**Files:**
- Modify: `tests/unit/test_eval_service.py`

**Step 1: Update tests**

Since `EvalService` is removed, rewrite `tests/unit/test_eval_service.py` to test the eval script functions instead:

```python
import json
import pytest
from unittest.mock import patch, mock_open
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
```

**Step 2: Run tests**

Run: `uv run pytest tests/unit/test_eval_service.py -v`
Expected: 4 PASSED

**Step 3: Run full lint + format**

```bash
uv run ruff check eval/run_eval.py scripts/seed_demo_data.py
uv run ruff format eval/run_eval.py scripts/seed_demo_data.py
```

**Step 4: Run all unit tests to ensure nothing is broken**

Run: `uv run pytest tests/unit -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add tests/unit/test_eval_service.py
git add -u
git commit -m "test: update eval tests for script-based flow"
```
