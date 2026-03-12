# Custom Evaluation Dataset Design

## Goal

Create a hand-verified evaluation dataset of ~70-80 questions across 8-12 landmark ML/AI papers, replacing reliance on FinanceBench for higher quality ground truth. Questions cover factual extraction, table/figure reasoning, and multi-hop reasoning.

## Paper Selection

8-12 papers chosen for structural diversity (tables, figures, equations, appendices). All freely available on arXiv.

| Paper | Structural challenge |
|-------|---------------------|
| Attention Is All You Need (2017) | Tables, figures, equations |
| BERT (Devlin 2019) | Multi-section ablation tables |
| ResNet (He 2015) | Heavy on figures/diagrams |
| GPT-2 (Radford 2019) | Long tables in appendix |
| Adam Optimizer (Kingma 2015) | Equations, pseudocode |
| Batch Normalization (Ioffe 2015) | Algorithm boxes, tables |
| YOLO (Redmon 2016) | Comparison tables, figures |
| GAN (Goodfellow 2014) | Math-heavy, fewer tables |
| Dropout (Srivastava 2014) | Many experiment tables |
| LoRA (Hu 2022) | Dense ablation tables |

The final list is flexible — swap papers as needed, targeting 8-12 total.

## Question Types & Distribution

Per paper: ~8 questions.

- **3-4 factual extraction** — short, exact answers from a single paragraph/sentence
- **2-3 table/figure reasoning** — requires reading or comparing values in tables/figures
- **1-2 multi-hop reasoning** — requires combining information from 2+ sections

Total target: ~70-80 questions.

## Q&A Generation Workflow

1. Ingest each paper into the pipeline via `IngestionService`
2. Feed full paper text to Claude/GPT-4o with structured prompts (one per question type)
3. Manually verify each Q&A against the actual PDF — correct, discard, or rewrite

### Generation Prompts

Three separate prompts:

- **Factual prompt** — asks for questions with short exact answers extractable from a single paragraph. LLM must cite the exact text span.
- **Table/figure prompt** — points at each table/figure, asks questions requiring reading or comparing specific values. Answer must reference specific cells/data points.
- **Multi-hop prompt** — asks for questions requiring info from 2+ sections. LLM must specify which sections are combined and why a single section is insufficient.

### Quality Control Rules

- Discard questions answerable from the abstract alone
- Discard questions with ambiguous answers or requiring outside knowledge
- Verify table questions reference actual table content (LLMs hallucinate table values)
- For multi-hop, confirm both source sections are actually needed

## Dataset Format

File: `eval/datasets/custom_dataset.json`

```json
{
  "dataset": "docmind-custom",
  "papers": [
    {
      "paper_id": "attention-is-all-you-need",
      "title": "Attention Is All You Need",
      "arxiv_url": "https://arxiv.org/abs/1706.03762",
      "filename": "attention_is_all_you_need.pdf"
    }
  ],
  "questions": [
    {
      "id": "att-f1",
      "paper_id": "attention-is-all-you-need",
      "question": "What was the BLEU score of the big Transformer on EN-DE WMT 2014?",
      "answer": "28.4",
      "type": "factual",
      "source_section": "Table 2",
      "source_page": 8,
      "difficulty": "easy"
    },
    {
      "id": "att-t1",
      "paper_id": "attention-is-all-you-need",
      "question": "How many parameters does the Transformer (big) model have compared to the base model?",
      "answer": "The big model has 213M parameters compared to 65M for the base model",
      "type": "table_reasoning",
      "source_section": "Table 3",
      "source_page": 9,
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

### Key Fields

- `type`: `factual` | `table_reasoning` | `multi_hop` — enables per-type metric breakdown
- `source_section` + `source_page`: ground truth for retrieval evaluation
- `difficulty`: `easy` | `medium` | `hard` — for analysis breakdown

## Seeding Script

New script: `scripts/seed_custom_eval.py`

- Reads paper list from `eval/datasets/custom_dataset.json`
- Downloads PDFs from arXiv URLs
- Ingests each via `IngestionService` (same pattern as `seed_demo_data.py`)
- Writes manifest to `eval/datasets/custom_manifest.json` with `paper_id` → `doc_id` mappings

Triggered via `make seed-custom`.

## Eval Runner Integration

Extend `eval/run_eval.py` to support `--dataset` flag:

```bash
make eval                          # default: financebench (unchanged)
make eval-custom                   # runs custom dataset
uv run python eval/run_eval.py --dataset custom
```

### Changes to `run_eval.py`

- `load_questions()` reads from local JSON when dataset is `custom` (instead of HuggingFace)
- Questions scoped to their `paper_id` → matched `doc_id` (per-paper scoping, not all docs)
- Same pipeline: RAGService.query() → RAGAS metrics → results JSON

### Metrics Output

Results saved to `eval/results/custom_results.json` with per-type breakdown:

```json
{
  "metrics": {
    "overall": { "faithfulness": 0.92, "context_recall": 0.78, "..." : "..." },
    "by_type": {
      "factual": { "faithfulness": 0.95, "context_recall": 0.88 },
      "table_reasoning": { "faithfulness": 0.90, "context_recall": 0.70 },
      "multi_hop": { "faithfulness": 0.85, "context_recall": 0.60 }
    }
  }
}
```

## File Layout

```
eval/
  datasets/
    custom_dataset.json          # Papers + Q&A pairs
    custom_manifest.json         # Generated: paper_id → doc_id mapping
  results/
    custom_results.json          # Eval output
  run_eval.py                    # Extended with --dataset flag
scripts/
  seed_custom_eval.py            # Download + ingest papers
Makefile                         # New targets: seed-custom, eval-custom
```
