from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...
