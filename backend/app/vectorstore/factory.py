from backend.app.core.config import settings
from backend.app.pipeline.base.vectorstore import BaseVectorStore


class VectorStoreFactory:

    @staticmethod
    def create(strategy: str | None = None) -> BaseVectorStore:
        strategy = strategy or settings.vectorstore_strategy

        if strategy == "elasticsearch":
            from backend.app.vectorstore.elasticsearch_store import ElasticsearchStore
            return ElasticsearchStore()

        raise ValueError(f"Unknown vectorstore strategy: {strategy}")
