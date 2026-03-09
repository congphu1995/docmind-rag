# Eval Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the RAG evaluation pipeline work end-to-end: seed FinanceBench docs, fix RAGAS v0.4 API, scope eval to seeded docs, update notebooks, produce committed result JSON.

**Architecture:** Pre-seed 5 FinanceBench 10-K PDFs under a system "eval" user. EvalService loads the seed manifest to get `doc_ids`, passes them in `ChatRequest` so retrieval is scoped. RAGAS v0.4.3 class-based API computes faithfulness, answer relevancy, and context recall. Notebooks call the HTTP API, poll for results, and save JSON.

**Tech Stack:** RAGAS v0.4.3, langchain-openai (for RAGAS LLM/embeddings), HuggingFace datasets, httpx (notebooks)

> **Depends on:** Ingestion pipeline, RAGService, auth system, Celery worker — all already built.

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
- Modify: `Makefile`

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

**Step 2: Update Makefile `seed` target**

The existing `make seed` already points to `scripts/seed_demo_data.py` — no change needed.

**Step 3: Run to verify it starts**

Run: `uv run python -c "from scripts.seed_demo_data import FINANCEBENCH_DOCS; print(len(FINANCEBENCH_DOCS))"`
Expected: `5`

**Step 4: Commit**

```bash
git add scripts/seed_demo_data.py
git commit -m "feat: seed script — eval user + FinanceBench PDF download + ingest"
```

---

## Task 3: Update RAGAS Metrics to v0.4.3 API

**Files:**
- Modify: `backend/app/services/eval.py`
- Modify: `tests/unit/test_eval_service.py`

**Step 1: Write the test for the new RAGAS integration**

Add to `tests/unit/test_eval_service.py`:

```python
async def test_compute_ragas_builds_correct_samples():
    """Verify RAGAS sample construction uses correct field names."""
    service = EvalService.__new__(EvalService)
    results = [
        {
            "question": "What was revenue?",
            "generated_answer": "Revenue was $100B.",
            "ground_truth": "$100B",
            "contexts": ["Revenue section: total revenue was $100B."],
        }
    ]
    # Should not raise — validates sample construction
    # We mock evaluate() to avoid actual LLM calls
    with patch("backend.app.services.eval.evaluate") as mock_eval:
        mock_eval.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_recall": 0.8,
        }
        metrics = await service._compute_ragas_metrics(results)
        assert metrics["faithfulness"] == 0.9
        assert metrics["answer_relevancy"] == 0.85
        assert metrics["context_recall"] == 0.8
        mock_eval.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_eval_service.py::test_compute_ragas_builds_correct_samples -v`
Expected: FAIL — import path or API mismatch

**Step 3: Rewrite `_compute_ragas_metrics` in `backend/app/services/eval.py`**

Replace the `_compute_ragas_metrics` method (lines 129-163) with:

```python
    async def _compute_ragas_metrics(self, results: list[dict]) -> dict:
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
                logger.warning("ragas_no_valid_samples")
                return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}

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
            logger.warning("ragas_metrics_failed", error=str(e))
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0,
            }
```

Also update the imports at the top of the file — remove unused `from datasets import Dataset` if present. The top-level imports stay the same (ragas is imported inside the method to avoid startup failures if not installed).

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_eval_service.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add backend/app/services/eval.py tests/unit/test_eval_service.py
git commit -m "fix: update RAGAS to v0.4.3 class-based API (SingleTurnSample, EvaluationDataset)"
```

---

## Task 4: Scope EvalService to Seeded Documents

**Files:**
- Modify: `backend/app/services/eval.py`
- Modify: `tests/unit/test_eval_service.py`

**Step 1: Write the test**

Add to `tests/unit/test_eval_service.py`:

```python
async def test_eval_loads_manifest():
    """Verify eval loads seed manifest and extracts doc_ids."""
    service = EvalService.__new__(EvalService)
    manifest = {
        "user_id": "test-user-123",
        "documents": [
            {"doc_name": "AAPL_10K.pdf", "doc_id": "doc-1", "hf_doc_name": "APPLE INC"},
            {"doc_name": "MSFT_10K.pdf", "doc_id": "doc-2", "hf_doc_name": "MICROSOFT CORP"},
        ],
    }
    with patch("builtins.open", mock_open(read_data=json.dumps(manifest))):
        with patch("pathlib.Path.exists", return_value=True):
            result = service._load_manifest()
            assert result["user_id"] == "test-user-123"
            assert len(result["doc_ids"]) == 2
            assert "doc-1" in result["doc_ids"]
```

Add these imports at the top of the test file:

```python
import json
from unittest.mock import mock_open
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_eval_service.py::test_eval_loads_manifest -v`
Expected: FAIL — `_load_manifest` not defined

**Step 3: Add manifest loading and doc scoping to EvalService**

In `backend/app/services/eval.py`, add after the imports:

```python
from pathlib import Path

MANIFEST_PATH = Path("eval/datasets/seed_manifest.json")
```

Add `_load_manifest` method and update `run_eval` to use it:

```python
    def _load_manifest(self) -> dict:
        """Load seed manifest to get doc_ids and user_id."""
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                "Seed manifest not found. Run `make seed` first to ingest eval documents."
            )
        data = json.loads(MANIFEST_PATH.read_text())
        return {
            "user_id": data["user_id"],
            "doc_ids": [d["doc_id"] for d in data["documents"]],
            "documents": data["documents"],
        }
```

Update the `run_eval` method — after `log.info("eval_start")`, add manifest loading. Update the `ChatRequest` construction in the results loop to include `doc_ids`:

Replace the ChatRequest creation block (lines 45-49) with:

```python
                request = ChatRequest(
                    question=q["question"],
                    llm="openai",
                    doc_ids=manifest["doc_ids"],
                    stream=False,
                )
```

The full updated `run_eval` method should start with:

```python
    async def run_eval(
        self,
        dataset: str,
        sample_size: int,
        config: dict,
    ) -> str:
        """Run evaluation. Returns run_id. Called from Celery task."""
        run_id = await self._create_run_record(dataset, sample_size, config)
        log = logger.bind(run_id=run_id, dataset=dataset)
        log.info("eval_start", sample_size=sample_size)

        try:
            # Load manifest for doc scoping
            manifest = self._load_manifest()
            log.info("eval_manifest_loaded", doc_ids=len(manifest["doc_ids"]))

            # Load dataset questions
            questions = await self._load_dataset(dataset, sample_size)
            log.info("eval_dataset_loaded", questions=len(questions))

            # Run RAG pipeline on each question
            results = []
            latencies = []
            for i, q in enumerate(questions):
                start = time.time()
                request = ChatRequest(
                    question=q["question"],
                    llm="openai",
                    doc_ids=manifest["doc_ids"],
                    stream=False,
                )
                # ... rest stays the same
```

Also add `import json` to the top imports.

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_eval_service.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add backend/app/services/eval.py tests/unit/test_eval_service.py
git commit -m "feat: scope eval to seeded docs via seed_manifest.json"
```

---

## Task 5: Add List Eval Runs Endpoint

**Files:**
- Modify: `backend/app/api/eval.py`
- Modify: `backend/app/services/eval.py`

**Step 1: Add `list_runs` method to EvalService**

Add to `backend/app/services/eval.py`:

```python
    async def list_runs(self) -> list[dict]:
        """List all eval runs, most recent first."""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(EvalRun).order_by(EvalRun.created_at.desc())
            )
            runs = result.scalars().all()
            return [
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "dataset": r.dataset,
                    "sample_size": r.sample_size,
                    "metrics": r.metrics,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in runs
            ]
```

**Step 2: Add the endpoint to `backend/app/api/eval.py`**

Add before the `get_eval_results` route:

```python
@router.get("/results")
async def list_eval_runs():
    """List all evaluation runs."""
    from backend.app.services.eval import EvalService

    service = EvalService()
    return await service.list_runs()
```

**Step 3: Verify import**

Run: `uv run python -c "from backend.app.api.eval import router; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add backend/app/api/eval.py backend/app/services/eval.py
git commit -m "feat: add GET /eval/results to list all eval runs"
```

---

## Task 6: Update Baseline Eval Notebook

**Files:**
- Rewrite: `eval/notebooks/01_baseline_eval.ipynb`

**Step 1: Rewrite the notebook**

Update `eval/notebooks/01_baseline_eval.ipynb` with these cells:

**Cell 0 (markdown):**
```markdown
# DocMind RAG — Baseline Evaluation

FinanceBench dataset (5 seeded 10-K filings, ~30 questions).
Default config: Parent-Child (800/150), text-embedding-3-small, GPT-4o.

**Prerequisites:** `make seed` must have been run first.
```

**Cell 1 (code):**
```python
import json
from pathlib import Path
import httpx

API_BASE = "http://localhost:8000/api/v1"

# Verify seed manifest exists
manifest_path = Path("../datasets/seed_manifest.json")
if not manifest_path.exists():
    raise FileNotFoundError("Run `make seed` first to ingest eval documents.")

manifest = json.loads(manifest_path.read_text())
print(f"Seeded docs: {len(manifest['documents'])}")
for doc in manifest["documents"]:
    print(f"  {doc['doc_name']} → {doc['doc_id']} ({doc.get('child_chunks', '?')} chunks)")
```

**Cell 2 (code):**
```python
# Start eval run
response = httpx.post(f"{API_BASE}/eval/run", json={
    "dataset": "financebench",
    "sample_size": 30,
}, timeout=30)
result = response.json()
print(f"Eval started: {result}")
run_id = result.get("run_id", "")
```

**Cell 3 (code):**
```python
# Poll for results
import time

data = {}
for attempt in range(60):
    r = httpx.get(f"{API_BASE}/eval/results/{run_id}", timeout=30)
    data = r.json()
    status = data.get("status", "unknown")
    if status in ("completed", "failed"):
        break
    print(f"[{attempt+1}/60] Status: {status}...")
    time.sleep(15)

print(f"\nFinal status: {data.get('status')}")
if data.get("error"):
    print(f"Error: {data['error']}")
```

**Cell 4 (code):**
```python
# Display results
if data.get("status") == "completed":
    metrics = data.get("metrics", {})

    print(f"{'='*50}")
    print(f"  DocMind RAG — Baseline Evaluation Results")
    print(f"{'='*50}")
    print(f"  Retrieval Hit Rate:  {metrics.get('retrieval_hit_rate', 0):.1%}")
    print(f"  Faithfulness:        {metrics.get('faithfulness', 0):.4f}")
    print(f"  Answer Relevancy:    {metrics.get('answer_relevancy', 0):.4f}")
    print(f"  Context Recall:      {metrics.get('context_recall', 0):.4f}")
    print(f"  Latency p95:         {metrics.get('latency_p95_ms', 0):.0f}ms")
    print(f"  Sample Size:         {metrics.get('sample_size', 0)}")
    print(f"{'='*50}")

    # Save results
    results_path = Path("../results/financebench_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {results_path}")
else:
    print(f"Eval failed: {data}")
```

**Step 2: Commit**

```bash
git add eval/notebooks/01_baseline_eval.ipynb
git commit -m "feat: update baseline eval notebook — manifest check, 30 samples, save results"
```

---

## Task 7: Update Ablation Notebook

**Files:**
- Rewrite: `eval/notebooks/02_chunking_ablation.ipynb`

**Step 1: Rewrite the notebook**

Update `eval/notebooks/02_chunking_ablation.ipynb` with these cells:

**Cell 0 (markdown):**
```markdown
# DocMind RAG — Chunking Ablation Study

Compares 4 configurations:
- **A**: Fixed-size (400 words, no parent-child, no enrichment)
- **B**: Parent-child (800/150, no enrichment)
- **C**: Parent-child + contextual enrichment
- **D**: Parent-child + enrichment + tables atomic (default)

Expected: D > C > B > A. Tables-atomic gap should be the largest single jump.

**Note:** Each config requires re-ingestion. This notebook takes ~30 min total.
```

**Cell 1 (code):**
```python
import json
import time
from pathlib import Path
import httpx

API_BASE = "http://localhost:8000/api/v1"
SAMPLE_SIZE = 30

configs = {
    "A_fixed_size": {
        "parent_max_words": 400,
        "child_max_words": 400,
        "enrichment": False,
        "tables_atomic": False,
    },
    "B_parent_child": {
        "parent_max_words": 800,
        "child_max_words": 150,
        "enrichment": False,
        "tables_atomic": False,
    },
    "C_enriched": {
        "parent_max_words": 800,
        "child_max_words": 150,
        "enrichment": True,
        "tables_atomic": False,
    },
    "D_full": {
        "parent_max_words": 800,
        "child_max_words": 150,
        "enrichment": True,
        "tables_atomic": True,
    },
}

run_ids = {}
for name, config in configs.items():
    print(f"Starting config: {name}")
    r = httpx.post(f"{API_BASE}/eval/run", json={
        "dataset": "financebench",
        "sample_size": SAMPLE_SIZE,
        "config": config,
    }, timeout=30)
    data = r.json()
    run_ids[name] = data.get("run_id", "")
    print(f"  run_id: {run_ids[name]}")
```

**Cell 2 (code):**
```python
# Poll all runs until complete
results = {}
for name, run_id in run_ids.items():
    print(f"Waiting for {name}...")
    for attempt in range(120):
        r = httpx.get(f"{API_BASE}/eval/results/{run_id}", timeout=30)
        data = r.json()
        if data.get("status") in ("completed", "failed"):
            results[name] = data
            status = data.get("status")
            print(f"  {name}: {status}")
            break
        if attempt % 4 == 0:
            print(f"  [{attempt}] {name}: {data.get('status', 'unknown')}...")
        time.sleep(15)
    else:
        results[name] = {"status": "timeout"}
        print(f"  {name}: TIMEOUT")
```

**Cell 3 (code):**
```python
# Comparison table
print(f"\n{'Config':<20} {'Hit Rate':<12} {'Faithful':<12} {'Relevancy':<12} {'Recall':<12} {'p95 (ms)':<10}")
print("=" * 78)

for name in ["A_fixed_size", "B_parent_child", "C_enriched", "D_full"]:
    data = results.get(name, {})
    m = data.get("metrics", {})
    if data.get("status") == "completed":
        print(
            f"{name:<20} "
            f"{m.get('retrieval_hit_rate', 0):<12.1%} "
            f"{m.get('faithfulness', 0):<12.4f} "
            f"{m.get('answer_relevancy', 0):<12.4f} "
            f"{m.get('context_recall', 0):<12.4f} "
            f"{m.get('latency_p95_ms', 0):<10.0f}"
        )
    else:
        print(f"{name:<20} {'FAILED':<12}")

# Save results
results_path = Path("../results/ablation_results.json")
results_path.parent.mkdir(parents=True, exist_ok=True)
results_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {results_path}")
```

**Step 2: Commit**

```bash
git add eval/notebooks/02_chunking_ablation.ipynb
git commit -m "feat: update ablation notebook — 4 configs, comparison table, save results"
```

---

## Task 8: Update Download Script + Eval README

**Files:**
- Modify: `eval/datasets/download_financebench.py`
- Modify: `eval/README.md`

**Step 1: Update download script to also save matched questions**

In `eval/datasets/download_financebench.py`, add after the main questions dump (after line 36), a section to filter questions relevant to our 5 seeded companies:

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

**Step 2: Update eval README**

Rewrite `eval/README.md`:

```markdown
# Evaluation

## Quick Start

```bash
# 1. Start services
docker compose up -d

# 2. Start backend + worker
make backend   # terminal 1
make worker    # terminal 2

# 3. Seed FinanceBench documents (downloads 5 10-K PDFs, ingests them)
make seed

# 4. Run baseline eval
# Open eval/notebooks/01_baseline_eval.ipynb in Jupyter

# 5. Run ablation study (optional, ~30 min)
# Open eval/notebooks/02_chunking_ablation.ipynb
```

## Datasets

- **FinanceBench**: QA pairs from real SEC filings (10-K, 10-Q)
- Source: `PatronusAI/financebench` on HuggingFace
- Download full dataset: `uv run python eval/datasets/download_financebench.py`
- Seed (5 filings): `make seed`

## Metrics

| Metric | Tool | Target |
|---|---|---|
| Retrieval Hit Rate | Custom | > 80% |
| Faithfulness | RAGAS | > 0.85 |
| Answer Relevancy | RAGAS | > 0.80 |
| Context Recall | RAGAS | > 0.75 |
| Latency p95 | Logged | < 3s |

## Reproduce

1. `make seed` to ingest eval documents
2. Run `eval/notebooks/01_baseline_eval.ipynb`
3. Results saved to `eval/results/financebench_results.json`

## Ablation

`eval/notebooks/02_chunking_ablation.ipynb` compares 4 chunking configs:
- A: Fixed-size 400 words
- B: Parent-child 800/150
- C: Parent-child + enrichment
- D: Parent-child + enrichment + tables atomic (default)

Results saved to `eval/results/ablation_results.json`
```

**Step 3: Commit**

```bash
git add eval/datasets/download_financebench.py eval/README.md
git commit -m "docs: update eval README + download script with matched questions"
```

---

## Task 9: Final Integration Test

**No new files — manual verification.**

**Step 1: Verify all imports**

Run:
```bash
uv run python -c "
from backend.app.services.eval import EvalService
from backend.app.api.eval import router
from scripts.seed_demo_data import FINANCEBENCH_DOCS
print('All imports OK')
"
```
Expected: `All imports OK`

**Step 2: Run all unit tests**

Run: `uv run pytest tests/unit/test_eval_service.py -v`
Expected: All PASSED

**Step 3: Run full lint**

Run: `uv run ruff check backend/app/services/eval.py backend/app/api/eval.py scripts/seed_demo_data.py`
Expected: No errors

**Step 4: Format**

Run: `uv run ruff format backend/app/services/eval.py backend/app/api/eval.py scripts/seed_demo_data.py`

**Step 5: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style: format eval files"
```
