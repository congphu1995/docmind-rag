# Eval Pipeline — Design Document

> End-to-end RAG evaluation: seed FinanceBench docs, fix RAGAS, run metrics, commit results.

| Field | Value |
|---|---|
| Author | Cong Phu Nguyen |
| Date | 2026-03-09 |
| Status | Approved |
| Scope | Seed script, RAGAS v0.4 fix, EvalService auth, notebooks, committed results |

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Document ingestion | Pre-seed 5 filings via script | Reproducible, manageable cost, tests full pipeline |
| RAGAS API | Update to v0.4.3 class-based API | Current code uses deprecated imports |
| Auth handling | System "eval" user | Tests real auth flow, data isolated by user_id |
| Dataset size | 5 filings, ~30 questions | Fast seed (~6 min), enough for meaningful metrics |
| Data isolation | Shared Qdrant + PostgreSQL, scoped by user_id | Auth already isolates users, no extra infra |
| UI | No eval page — notebooks + committed JSON only | Eval is for technical reviewers, not end users |

---

## Section 1: Seed Script

Rewrite `scripts/seed_demo_data.py`:
- Download 5 specific FinanceBench 10-K PDFs (deterministic set by company/year)
- Create system "eval" user (`eval@docmind.system`, random password)
- Ingest each PDF under that user via `IngestionService`
- Save `eval/datasets/seed_manifest.json`:
  ```json
  {
    "user_id": "uuid",
    "documents": [
      {"doc_name": "AAPL_10K_2023.pdf", "doc_id": "uuid"}
    ]
  }
  ```
- Expose via `make seed`

---

## Section 2: Fix RAGAS Integration

Update `EvalService._compute_ragas_metrics()` for RAGAS v0.4.3:

**Old API (deprecated):**
```python
from ragas.metrics import faithfulness, answer_relevancy, context_recall
evaluate(dataset, metrics=[faithfulness, ...])
```

**New API:**
```python
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas import EvaluationDataset, SingleTurnSample

samples = [SingleTurnSample(
    user_input=q["question"],
    response=q["generated_answer"],
    reference=q["ground_truth"],
    retrieved_contexts=q["contexts"],
)]
dataset = EvaluationDataset(samples=samples)

metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings), ContextRecall(llm=llm)]
result = evaluate(dataset=dataset, metrics=metrics)
```

- Use `langchain_openai.ChatOpenAI` and `langchain_openai.OpenAIEmbeddings` as LLM/embeddings for RAGAS
- Model: `gpt-4o-mini` for eval (cheap, fast)

---

## Section 3: Fix EvalService Auth + Doc Scoping

- `EvalService` loads `eval/datasets/seed_manifest.json` to get `doc_ids` and `user_id`
- Passes `doc_ids` in `ChatRequest` so retrieval is scoped to eval documents
- RAGService already accepts `doc_ids` — no changes needed there
- If manifest not found, raise clear error: "Run `make seed` first"

---

## Section 4: Eval API Tweaks

- `POST /eval/run` — no auth required (system operation)
- `GET /eval/results/{run_id}` — no changes
- Add `GET /eval/results` — list all eval runs (for notebooks)

---

## Section 5: Notebooks

**`01_baseline_eval.ipynb`:**
- Check seed manifest exists
- POST `/eval/run` with `dataset=financebench`, `sample_size=30`
- Poll results until complete
- Display metrics table
- Save to `eval/results/financebench_results.json`

**`02_chunking_ablation.ipynb`:**
- Run 4 configs (A/B/C/D) via config overrides:
  - A: fixed-size 400 words, no parent-child, no enrichment
  - B: parent-child 800/150, no enrichment
  - C: parent-child + enrichment
  - D: parent-child + enrichment + tables atomic (default)
- Compare metrics side-by-side
- Save to `eval/results/ablation_results.json`

---

## Section 6: Committed Results

- `eval/results/financebench_results.json` — baseline metrics
- `eval/results/ablation_results.json` — ablation comparison
- Both committed to git for reproducibility
- README eval table updated with actual numbers

---

## Target Metrics (from DESIGN.md)

| Metric | Tool | Target |
|---|---|---|
| Retrieval Hit Rate | Custom | > 80% |
| Faithfulness | RAGAS | > 0.85 |
| Answer Relevancy | RAGAS | > 0.80 |
| Context Recall | RAGAS | > 0.75 |
| Latency p95 | Logged | < 3s |
