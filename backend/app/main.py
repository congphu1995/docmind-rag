from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from backend.app.api import auth, chat, chunks, documents, health
from backend.app.core.database import create_tables
from backend.app.core.langfuse import configure_langfuse
from backend.app.core.logging import configure_logging
from backend.app.core.middleware import PrometheusMiddleware
from backend.app.vectorstore.factory import VectorStoreFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_langfuse()
    configure_logging()
    await create_tables()
    vectorstore = VectorStoreFactory.create()
    await vectorstore.initialize()
    yield


app = FastAPI(
    title="DocMind RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)

app.include_router(
    documents.router, prefix="/api/v1/documents", tags=["documents"]
)
app.include_router(
    chat.router, prefix="/api/v1/chat", tags=["chat"]
)
app.include_router(chunks.router, prefix="/api/v1", tags=["chunks"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])

# Mount /metrics for Prometheus scraping
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
