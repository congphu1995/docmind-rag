#!/usr/bin/env python3
"""
Run RAG evaluation against seeded FinanceBench documents.
Calls RAGService directly — no HTTP, no Celery.

Usage: uv run python eval/run_eval.py
Requires: `make seed` to have been run first.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env so OPENAI_API_KEY is available for langchain_openai

MANIFEST_PATH = Path("eval/datasets/seed_manifest.json")
RESULTS_PATH = Path("eval/results/financebench_results.json")
SAMPLE_SIZE = 30

CUSTOM_DATASET_PATH = Path("eval/datasets/custom_dataset.json")
CUSTOM_MANIFEST_PATH = Path("eval/datasets/custom_manifest.json")
CUSTOM_RESULTS_PATH = Path("eval/results/custom_results.json")


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Seed manifest not found. Run `make seed` first.")
    data = json.loads(MANIFEST_PATH.read_text())
    return {
        "user_id": data["user_id"],
        "doc_ids": [d["doc_id"] for d in data["documents"]],
        "documents": data["documents"],
    }


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


async def load_questions(manifest: dict, sample_size: int) -> list[dict]:
    from datasets import load_dataset

    hf_names = {d["hf_doc_name"].lower() for d in manifest["documents"]}

    ds = load_dataset("PatronusAI/financebench", split="train")

    matched = []
    for item in ds:
        doc_name = item.get("doc_name", "").lower()
        if any(name in doc_name for name in hf_names):
            matched.append(
                {
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                    "doc_name": item.get("doc_name", ""),
                }
            )
        if len(matched) >= sample_size:
            break

    if not matched:
        print("WARNING: No matched questions found. Using first N from dataset.")
        for item in ds.select(range(min(sample_size, len(ds)))):
            matched.append(
                {
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                    "doc_name": item.get("doc_name", ""),
                }
            )

    return matched


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


def calculate_hit_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.get("relevant_found"))
    return round(hits / len(results), 4)


async def compute_ragas_metrics(results: list[dict]) -> dict:
    try:
        import os
        import warnings

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import AnswerRelevancy, ContextRecall, Faithfulness

        api_key = os.environ.get("OPENAI_API_KEY", "")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

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

        import math

        def safe_round(val, digits=4):
            if isinstance(val, list):
                valid = [v for v in val if v is not None and not math.isnan(v)]
                val = sum(valid) / max(len(valid), 1) if valid else 0.0
            f = float(val) if val is not None else 0.0
            return round(f, digits) if not math.isnan(f) else 0.0

        return {
            "faithfulness": safe_round(ragas_result["faithfulness"]),
            "answer_relevancy": safe_round(ragas_result["answer_relevancy"]),
            "context_recall": safe_round(ragas_result["context_recall"]),
        }
    except Exception as e:
        print(f"WARNING: RAGAS metrics failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}


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


if __name__ == "__main__":
    asyncio.run(main())
