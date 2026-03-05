import asyncio

from backend.app.workers.celery_app import celery_app


@celery_app.task(name="eval.run", bind=True)
def run_eval_task(self, dataset: str, sample_size: int, config: dict):
    """Async eval task — bridges Celery sync → async."""
    from backend.app.services.eval import EvalService

    service = EvalService()
    run_id = asyncio.run(service.run_eval(dataset, sample_size, config))
    return {"run_id": run_id, "status": "completed"}
