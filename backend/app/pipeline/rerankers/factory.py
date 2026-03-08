from backend.app.core.config import settings
from backend.app.pipeline.base.reranker import BaseReranker
from backend.app.pipeline.rerankers.identity_reranker import IdentityReranker


class RerankerFactory:

    @staticmethod
    def create(strategy: str = None) -> BaseReranker:
        strategy = strategy or settings.reranker_strategy

        if strategy == "cohere":
            from backend.app.pipeline.rerankers.cohere_reranker import CohereReranker
            return CohereReranker(api_key=settings.cohere_api_key)

        if strategy == "identity":
            return IdentityReranker()

        raise ValueError(
            f"Unknown reranker: {strategy}. Choose: identity, cohere"
        )
