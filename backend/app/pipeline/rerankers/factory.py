from backend.app.pipeline.base.reranker import BaseReranker
from backend.app.pipeline.rerankers.identity_reranker import IdentityReranker


class RerankerFactory:

    @staticmethod
    def create(strategy: str = "identity") -> BaseReranker:
        rerankers = {
            "identity": IdentityReranker,
        }
        if strategy not in rerankers:
            raise ValueError(
                f"Unknown reranker: {strategy}. Choose: {list(rerankers.keys())}"
            )
        return rerankers[strategy]()
