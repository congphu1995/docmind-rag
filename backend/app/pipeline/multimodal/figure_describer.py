"""
Describes figures/images using GPT-4o Vision.
Produces a text description suitable for embedding + retrieval.
"""
from backend.app.core.logging import logger
from backend.app.pipeline.base.llm_client import BaseLLMClient
from backend.app.pipeline.prompts import FIGURE_DESCRIPTION_PROMPT


class FigureDescriber:

    def __init__(self, llm: BaseLLMClient):
        self._llm = llm

    async def describe(
        self,
        image_b64: str,
        doc_context: str = "",
    ) -> str:
        """Generate text description of a figure from base64 image."""
        if not image_b64:
            raise ValueError("image_b64 cannot be empty")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": FIGURE_DESCRIPTION_PROMPT.format(
                            doc_context=doc_context
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        description = await self._llm.complete(messages, max_tokens=300)
        logger.info(
            "figure_described",
            description_len=len(description),
            doc_context=doc_context[:50],
        )
        return description.strip()

    async def describe_batch(
        self,
        figures: list[dict],
        doc_context: str = "",
    ) -> list[str]:
        """Describe multiple figures. Each dict has 'image_b64' key."""
        results = []
        for fig in figures:
            desc = await self.describe(
                image_b64=fig["image_b64"],
                doc_context=doc_context,
            )
            results.append(desc)
        return results
