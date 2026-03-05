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
