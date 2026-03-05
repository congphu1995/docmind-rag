import asyncio

from backend.app.core.logging import logger
from backend.app.services.ingestion import IngestionService
from backend.app.workers.celery_app import celery_app


@celery_app.task(bind=True, name="ingest_document")
def ingest_document_task(
    self,
    file_path: str,
    doc_name: str,
    language: str = "en",
    parser_strategy: str = "auto",
):
    """Async ingestion task — runs in Celery worker, not HTTP request."""
    try:
        service = IngestionService()
        result = asyncio.run(
            service.ingest(file_path, doc_name, language, parser_strategy)
        )
        return result
    except Exception as e:
        logger.error(
            "celery_ingest_failed", task_id=self.request.id, error=str(e)
        )
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
