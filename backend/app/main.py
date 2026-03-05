from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import chat, documents, eval, health
from backend.app.core.database import create_tables
from backend.app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await create_tables()
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

app.include_router(
    documents.router, prefix="/api/v1/documents", tags=["documents"]
)
app.include_router(
    chat.router, prefix="/api/v1/chat", tags=["chat"]
)
app.include_router(
    eval.router, prefix="/api/v1/eval", tags=["eval"]
)
app.include_router(health.router, prefix="/api/v1", tags=["health"])
