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

## FinanceBench Baseline Results

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

---

## Custom Dataset (ML/AI Papers)

A hand-verified evaluation dataset of 82 questions across 10 landmark ML/AI papers. Covers factual extraction, table/figure reasoning, and multi-hop reasoning.

### Quick Start

```bash
# 1. Start services
docker compose up -d

# 2. Seed papers (downloads 10 PDFs from arXiv, ingests them)
make seed-custom

# 3. Run evaluation
make eval-custom

# 4. View results
cat eval/results/custom_results.json
```

### Papers

| Paper | Year | Structural Challenge |
|---|---|---|
| Attention Is All You Need | 2017 | Tables, figures, equations |
| BERT | 2019 | Multi-section ablation tables |
| ResNet | 2015 | Heavy on figures/diagrams |
| GPT-2 | 2019 | Long tables in appendix |
| Adam Optimizer | 2015 | Equations, pseudocode |
| Batch Normalization | 2015 | Algorithm boxes, tables |
| YOLO | 2016 | Comparison tables, figures |
| GAN | 2014 | Math-heavy, fewer tables |
| Dropout | 2014 | Many experiment tables |
| LoRA | 2022 | Dense ablation tables |

### Question Types

- **Factual** (39q) — short, exact answers from a single paragraph/sentence
- **Table Reasoning** (23q) — requires reading or comparing values in tables/figures
- **Multi-hop** (20q) — requires combining information from 2+ sections

### Custom Dataset Results

**Overall:**

| Metric | Score | Target | Status |
|---|---|---|---|
| Retrieval Hit Rate | **95.1%** | > 80% | PASS |
| Faithfulness | **0.9478** | > 0.85 | PASS |
| Answer Relevancy | **0.8217** | > 0.80 | PASS |
| Context Recall | **0.8899** | > 0.75 | PASS |
| Latency p95 | **19,150ms** | < 3,000ms | BELOW |
| Sample Size | 82 | ~80 | — |

**Per-type breakdown:**

| Type | Hit Rate | Faithfulness | Relevancy | Recall |
|---|---|---|---|---|
| Factual | 94.9% | 0.988 | 0.817 | 0.921 |
| Table Reasoning | 95.7% | 0.940 | 0.755 | 0.870 |
| Multi-hop | 95.0% | 0.930 | 0.895 | 0.938 |

**Notes:**
- All core metrics pass targets — significant improvement over FinanceBench (9 → 82 questions).
- Retrieval is strong across all question types (~95% hit rate).
- Faithfulness is highest for factual questions (0.988) — the model sticks closely to retrieved context for simple lookups.
- Table reasoning has lowest relevancy (0.755) — table data is harder to extract and compare accurately.
- Multi-hop has surprisingly high recall (0.938) — the pipeline effectively combines info from multiple sections.
- Latency is still dominated by LLM generation (gpt-4o); retrieval itself is fast.

### Adding Questions

Edit `eval/datasets/custom_dataset.json`. Each question follows this format:

```json
{
  "id": "att-f1",
  "paper_id": "attention-is-all-you-need",
  "question": "What was the BLEU score of the big Transformer on EN-DE WMT 2014?",
  "answer": "28.4",
  "type": "factual",
  "source_section": "Table 2",
  "source_page": [8],
  "difficulty": "easy"
}
```

ID convention: `{paper_slug}-{type_initial}{sequence}` (e.g., `bert-t2`, `resnet-m1`).

After adding questions, bump the `version` field and re-run `make eval-custom`.

---

## Reproduce

**FinanceBench:**
1. `make seed` → `make eval`
2. Results: `eval/results/financebench_results.json`
3. Notebook: `eval/notebooks/01_baseline_eval.ipynb`

**Custom Dataset:**
1. `make seed-custom` → `make eval-custom`
2. Results: `eval/results/custom_results.json`
