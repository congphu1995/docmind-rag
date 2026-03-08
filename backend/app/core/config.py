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
    parent_chunk_size: int = 800      # target when splitting oversized sections
    parent_min_words: int = 200       # merge small sections below this
    parent_max_words: int = 1200      # split sections above this
    child_chunk_size: int = 150       # target when splitting oversized paragraphs
    child_min_words: int = 50         # merge small paragraphs below this
    child_max_words: int = 250        # split paragraphs above this
    chunk_overlap: int = 15           # kept for backward compat, unused by new chunker

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

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
