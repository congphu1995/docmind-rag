# Evaluation

## Quick Start

```bash
# 1. Start services
docker compose up -d

# 2. Seed with FinanceBench documents
make seed

# 3. Run baseline eval
# Open eval/notebooks/01_baseline_eval.ipynb in Jupyter

# 4. Run ablation study
# Open eval/notebooks/02_chunking_ablation.ipynb
```

## Datasets

- **FinanceBench**: 10K+ QA pairs from real SEC filings (10-K, 10-Q)
- Download: `uv run python eval/datasets/download_financebench.py`

## Metrics

| Metric | Tool | Target |
|---|---|---|
| Retrieval Hit Rate | Custom | > 80% |
| Faithfulness | RAGAS | > 0.85 |
| Answer Relevancy | RAGAS | > 0.80 |
| Context Recall | RAGAS | > 0.75 |
| Latency p95 | Logged | < 3s |

## Reproduce

Results are committed to `results/financebench_results.json`.
