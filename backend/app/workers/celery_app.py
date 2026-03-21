from celery import Celery

from backend.app.core.config import settings

celery_app = Celery(
    "docmind",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,
    task_soft_time_limit=600,
    task_time_limit=900,
    include=[
        "backend.app.workers.ingest_tasks",
    ],
)
