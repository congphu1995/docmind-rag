from fastapi import APIRouter

from backend.app.core.logging import logger
from backend.app.schemas.eval import EvalRunRequest, EvalRunResponse
from backend.app.workers.eval_tasks import run_eval_task

router = APIRouter()


@router.post("/run", response_model=EvalRunResponse)
async def start_eval(request: EvalRunRequest):
    """Start an async evaluation run."""
    logger.info(
        "eval_requested",
        dataset=request.dataset,
        sample_size=request.sample_size,
    )

    task = run_eval_task.delay(
        dataset=request.dataset,
        sample_size=request.sample_size,
        config=request.config,
    )

    return EvalRunResponse(
        run_id=task.id,
        status="running",
        dataset=request.dataset,
        sample_size=request.sample_size,
    )


@router.get("/results/{run_id}")
async def get_eval_results(run_id: str):
    """Get evaluation results by run_id."""
    from backend.app.services.eval import EvalService

    service = EvalService()
    result = await service.get_result(run_id)
    if not result:
        # Check Celery task status as fallback
        task = run_eval_task.AsyncResult(run_id)
        if task.state == "PENDING":
            return {"run_id": run_id, "status": "pending"}
        elif task.state == "FAILURE":
            return {"run_id": run_id, "status": "failed", "error": str(task.result)}
        return {"run_id": run_id, "status": task.state.lower()}
    return result
