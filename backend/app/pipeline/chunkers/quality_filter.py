import re

from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk


class QualityFilter:
    MIN_WORDS = 15
    MIN_ALPHA_RATIO = 0.4
    MAX_NUMERIC_LINE_RATIO = 0.8

    def filter(self, chunks: list[Chunk]) -> list[Chunk]:
        """Remove noise chunks before indexing."""
        before = len(chunks)
        filtered = [c for c in chunks if self._is_quality(c)]
        removed = before - len(filtered)

        if removed > 0:
            logger.info("quality_filter", removed=removed, kept=len(filtered))

        return filtered

    def _is_quality(self, chunk: Chunk) -> bool:
        if chunk.is_parent or chunk.type != "text":
            return True

        text = chunk.content_raw
        words = text.split()

        if len(words) < self.MIN_WORDS:
            return False

        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars / max(len(text), 1) < self.MIN_ALPHA_RATIO:
            return False

        if re.search(r"(.)\1{4,}", text):
            return False

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        numeric_lines = sum(
            1 for line in lines if re.match(r"^[\d\s\.,\-\+%$€£]+$", line)
        )
        if lines and numeric_lines / len(lines) > self.MAX_NUMERIC_LINE_RATIO:
            return False

        return True
