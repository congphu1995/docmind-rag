from abc import ABC, abstractmethod
from typing import Optional


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query_vector: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]: ...
