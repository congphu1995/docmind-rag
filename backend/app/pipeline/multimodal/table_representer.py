"""
Creates 3 representations of a table:
- natural_language: for embedding (matches how users phrase questions)
- markdown: for LLM context (easier to reason about structure)
- html: for UI rendering (original fidelity)
"""
import re
from dataclasses import dataclass

from backend.app.core.logging import logger
from backend.app.pipeline.base.llm_client import BaseLLMClient
from backend.app.pipeline.prompts import TABLE_NL_PROMPT


@dataclass
class TableRepresentations:
    natural_language: str
    markdown: str
    html: str


class TableRepresenter:

    def __init__(self, llm: BaseLLMClient):
        self._llm = llm

    async def represent(
        self,
        table_html: str,
        section_context: str = "",
    ) -> TableRepresentations:
        """Generate all 3 representations from HTML table."""
        if not table_html.strip():
            raise ValueError("table_html cannot be empty")

        markdown = self._html_to_markdown(table_html)

        nl = await self._generate_nl(markdown, section_context)

        logger.info(
            "table_represented",
            markdown_len=len(markdown),
            nl_len=len(nl),
        )

        return TableRepresentations(
            natural_language=nl,
            markdown=markdown,
            html=table_html,
        )

    async def _generate_nl(
        self, table_markdown: str, section_context: str
    ) -> str:
        """Use LLM to convert table to natural language description."""
        prompt = TABLE_NL_PROMPT.format(
            table=table_markdown,
            section_context=section_context,
        )
        return await self._llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML table to markdown table."""
        html = re.sub(r"\s+", " ", html.strip())

        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
        if not rows:
            return html

        table_data = []
        for row in rows:
            cells = re.findall(
                r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", row, re.DOTALL
            )
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            table_data.append(cells)

        if not table_data:
            return html

        col_count = max(len(row) for row in table_data)
        for row in table_data:
            while len(row) < col_count:
                row.append("")

        lines = []
        header = table_data[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")

        for row in table_data[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
