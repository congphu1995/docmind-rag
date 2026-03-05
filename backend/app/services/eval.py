"""
Runs RAGAS evaluation against a dataset.
- Creates eval run record in PostgreSQL
- Retrieves + generates answers for sample questions
- Computes RAGAS metrics (faithfulness, answer_relevancy, context_recall)
- Stores results
"""
import time
import uuid
from datetime import datetime

from backend.app.core.database import AsyncSessionLocal
from backend.app.core.logging import logger
from backend.app.models.eval import EvalRun
from backend.app.schemas.chat import ChatRequest
from backend.app.services.rag import RAGService


class EvalService:

    def __init__(self):
        self._rag = RAGService()

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
                    stream=False,
                )
                response = await self._rag.query(request)
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
                })

                if (i + 1) % 10 == 0:
                    log.info("eval_progress", completed=i + 1, total=len(questions))

            # Compute metrics
            metrics = await self._compute_ragas_metrics(results)
            metrics["retrieval_hit_rate"] = self._calculate_hit_rate(results)
            metrics["sample_size"] = len(results)

            # Compute latency p95
            latencies.sort()
            p95_idx = int(len(latencies) * 0.95)
            metrics["latency_p95_ms"] = round(latencies[p95_idx] if latencies else 0, 1)

            # Store results
            await self._update_run(run_id, "completed", metrics)
            log.info("eval_done", **metrics)
            return run_id

        except Exception as e:
            log.error("eval_failed", error=str(e))
            await self._update_run(run_id, "failed", {}, str(e))
            raise

    async def _load_dataset(
        self, dataset: str, sample_size: int
    ) -> list[dict]:
        """Load evaluation questions from dataset."""
        if dataset == "financebench":
            return await self._load_financebench(sample_size)
        raise ValueError(f"Unknown dataset: {dataset}")

    async def _load_financebench(self, sample_size: int) -> list[dict]:
        """Load FinanceBench questions from HuggingFace datasets."""
        try:
            from datasets import load_dataset

            ds = load_dataset(
                "PatronusAI/financebench", split="train"
            )
            questions = []
            for item in ds.select(range(min(sample_size, len(ds)))):
                questions.append({
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                    "doc_name": item.get("doc_name", ""),
                })
            return questions
        except Exception as e:
            logger.warning("financebench_load_failed", error=str(e))
            return self._synthetic_questions(sample_size)

    def _synthetic_questions(self, n: int) -> list[dict]:
        """Fallback synthetic questions if dataset unavailable."""
        return [
            {
                "question": f"What was the total revenue in the most recent fiscal year? (sample {i})",
                "answer": "",
            }
            for i in range(n)
        ]

    async def _compute_ragas_metrics(self, results: list[dict]) -> dict:
        """Compute RAGAS metrics: faithfulness, answer_relevancy, context_recall."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_recall,
                faithfulness,
            )

            eval_dataset = Dataset.from_dict({
                "question": [r["question"] for r in results],
                "answer": [r["generated_answer"] for r in results],
                "contexts": [r["contexts"] for r in results],
                "ground_truth": [r["ground_truth"] for r in results],
            })

            ragas_result = evaluate(
                eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_recall],
            )

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

    def _calculate_hit_rate(self, results: list[dict]) -> float:
        if not results:
            return 0.0
        hits = sum(1 for r in results if r.get("relevant_found"))
        return round(hits / len(results), 4)

    async def _create_run_record(
        self, dataset: str, sample_size: int, config: dict
    ) -> str:
        run_id = str(uuid.uuid4())[:12]
        async with AsyncSessionLocal() as session:
            run = EvalRun(
                run_id=run_id,
                dataset=dataset,
                sample_size=sample_size,
                status="running",
                config=config,
            )
            session.add(run)
            await session.commit()
        return run_id

    async def _update_run(
        self,
        run_id: str,
        status: str,
        metrics: dict,
        error: str = "",
    ):
        async with AsyncSessionLocal() as session:
            from sqlalchemy import update

            await session.execute(
                update(EvalRun)
                .where(EvalRun.run_id == run_id)
                .values(
                    status=status,
                    metrics=metrics,
                    error=error,
                    completed_at=datetime.utcnow() if status != "running" else None,
                )
            )
            await session.commit()

    async def get_result(self, run_id: str) -> dict | None:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(EvalRun).where(EvalRun.run_id == run_id)
            )
            run = result.scalar_one_or_none()
            if not run:
                return None
            return {
                "run_id": run.run_id,
                "status": run.status,
                "dataset": run.dataset,
                "metrics": run.metrics,
                "error": run.error,
            }
