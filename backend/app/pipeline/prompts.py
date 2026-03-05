"""
All prompts used in the pipeline.
Never put prompts inline in business logic.
"""

ENRICHMENT_PROMPT = """\
Given this document context, write 1-2 sentences that situate the chunk \
within the document. Be specific about the section and topic. \
Output ONLY the situating sentences — no preamble, no explanation.

Document title: {doc_title}
Document type: {doc_type}
Current section: {section}

Chunk text:
{chunk_text}"""

METADATA_EXTRACTION_PROMPT = """\
Extract metadata from this document excerpt.

Document:
{sample_text}"""

FIGURE_DESCRIPTION_PROMPT = """\
Describe this figure from a document in detail.
Focus on: what type of chart/diagram it is, key data points, trends, labels, and axes.
Be specific with numbers when visible.
Output a paragraph suitable for document search — someone searching for this information should find it.

Document context: {doc_context}"""
