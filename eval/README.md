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
