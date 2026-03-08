from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # App
    app_name: str = "DocMind RAG"
    debug: bool = False

    # LLM
    openai_api_key: str
    anthropic_api_key: str = ""
    default_llm: Literal["claude", "openai"] = "openai"

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Parser
    parser_strategy: Literal["docling", "pymupdf", "auto"] = "auto"

    # Chunker (word counts)
    parent_chunk_size: int = 800
    child_chunk_size: int = 150
    chunk_overlap: int = 15

    # Retrieval
    retrieval_top_k: int = 20
    retrieval_top_n: int = 5
    retrieval_score_threshold: float = 0.4
    retrieval_quality_threshold: float = 0.6

    # Reranker
    reranker_strategy: str = "identity"
    cohere_api_key: str = ""
    reranker_top_n: int = 5

    # HyDE
    hyde_model: str = "gpt-4o-mini"
    hyde_max_tokens: int = 150

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "docmind_chunks"

    # PostgreSQL
    postgres_url: str = "postgresql+asyncpg://docmind:docmind@localhost:5432/docmind"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "docmind-files"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3001"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
