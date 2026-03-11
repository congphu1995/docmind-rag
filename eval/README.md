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

## Test Set

Source: [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) — 150 questions across SEC filings.

**Seeded documents** (5 10-K annual reports):

| Document | Pages | Elements | Chunks Indexed |
|---|---|---|---|
| AMAZON_2019_10K | 83 | 3,088 | 648 |
| COCACOLA_2022_10K | 183 | 5,918 | 1,518 |
| MICROSOFT_2023_10K | 116 | 4,334 | 824 |
| NIKE_2023_10K | 106 | 3,879 | 848 |
| PEPSICO_2022_10K | 503 | 14,088 | 3,086 |

9 questions from the dataset match these documents (factual, analytical, and multi-hop types).

## Baseline Results

| Metric | Score | Target | Status |
|---|---|---|---|
| Retrieval Hit Rate | **100.0%** | > 80% | PASS |
| Faithfulness | **0.9841** | > 0.85 | PASS |
| Answer Relevancy | **0.5873** | > 0.80 | BELOW |
| Context Recall | **0.4444** | > 0.75 | BELOW |
| Latency p95 | **22,985ms** | < 3,000ms | BELOW |
| Sample Size | 9 | ~30 | — |

**Notes:**
- Retrieval and faithfulness exceed targets — the pipeline finds relevant chunks and generates grounded answers.
- Answer relevancy is below target due to complex financial calculation questions (EBITDA, payout ratios) where the model hedges or provides partial answers.
- Context recall is below target — some ground truth facts aren't covered by the top-5 retrieved chunks (could improve with CohereReranker or more chunks).
- Latency is dominated by LLM generation (gpt-4o), not retrieval.

## Reproduce

1. `make seed` — ingest eval documents
2. `make eval` — run evaluation
3. Results: `eval/results/financebench_results.json`
4. Notebook: `eval/notebooks/01_baseline_eval.ipynb`
