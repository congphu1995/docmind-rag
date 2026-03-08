# backend/app/core/metrics.py
from prometheus_client import Counter, Gauge, Histogram

# LLM metrics
LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    labelnames=["provider", "model"],
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    labelnames=["provider", "model", "type"],
)

LLM_COST_TOTAL = Counter(
    "llm_cost_dollars_total",
    "Estimated LLM cost in USD",
    labelnames=["provider", "model"],
)

# HTTP metrics
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    labelnames=["method", "endpoint", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Pipeline metrics
INGESTION_STAGE_DURATION = Histogram(
    "ingestion_stage_duration_seconds",
    "Duration of each ingestion stage",
    labelnames=["stage"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

INGESTION_DOCUMENTS_TOTAL = Counter(
    "ingestion_documents_total",
    "Total documents ingested",
    labelnames=["status"],
)

CHUNKS_CREATED_TOTAL = Counter(
    "chunks_created_total",
    "Total chunks created",
    labelnames=["type"],
)

# Retrieval metrics
RETRIEVAL_DURATION = Histogram(
    "retrieval_duration_seconds",
    "Vector search latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

RERANKER_DURATION = Histogram(
    "reranker_duration_seconds",
    "Reranker latency",
    labelnames=["strategy"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

# Celery metrics
CELERY_TASKS_TOTAL = Counter(
    "celery_tasks_total",
    "Total Celery tasks",
    labelnames=["task", "status"],
)

CELERY_QUEUE_DEPTH = Gauge(
    "celery_queue_depth",
    "Current Celery queue depth",
)
