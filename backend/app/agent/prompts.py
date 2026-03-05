"""
All prompts for the agentic query pipeline.
Never put prompts inline in node logic.
Versioned with rationale — see comments.
"""

# ── Query Analysis ─────────────────────────────────────────────
# Used by: query_analyzer node
# Model: gpt-4o-mini (structured output)
# Rationale: One fast LLM call classifies the query so downstream
# nodes can make conditional decisions (HyDE, decomposition, routing).

QUERY_ANALYSIS_SYSTEM = """\
You are a query classifier for a document retrieval system.
Classify the user's query and extract metadata."""

QUERY_ANALYSIS_PROMPT = """\
Analyze this query and classify it.

Query: {query}

Query types:
- factual: asks for a specific fact, number, date, or definition
- analytical: asks for comparison, analysis, reasoning, or explanation
- tabular: asks about data in tables, charts, or structured numbers
- multi_hop: requires combining information from multiple sections or documents
- general: general knowledge question not about any uploaded document
- greeting: hello, hi, thanks, etc.

Respond with the classification."""


# ── Query Rewrite ──────────────────────────────────────────────
# Used by: query_rewriter node
# Model: gpt-4o-mini
# Rationale: Expand abbreviations, resolve references from history,
# make the query self-contained for embedding search.

QUERY_REWRITE_PROMPT = """\
Rewrite this query for optimal document retrieval search.
Expand abbreviations, resolve pronouns, make it self-contained.
Output ONLY the rewritten query — no preamble.

Original query: {query}
Conversation context: {context}"""


# ── HyDE (Hypothetical Document Embedding) ────────────────────
# Used by: query_rewriter node (conditional — see section 3.6)
# Model: gpt-4o-mini, max_tokens=150
# Rationale: Generate a hypothetical answer to improve embedding
# similarity. Only for analytical/multi_hop/vague queries.
# Quality doesn't matter — just needs to land near the right
# embedding neighborhood. 150 tokens keeps latency ~300ms.

HYDE_PROMPT = """\
Write a short passage that would answer this question,
as if it appeared in a professional document.
Be specific with terminology. ~100 words.

Question: {query}"""


# ── Decomposer ─────────────────────────────────────────────────
# Used by: decomposer node (multi_hop only)
# Model: gpt-4o-mini
# Rationale: Break complex multi-hop questions into 2-3 simpler
# sub-questions that can each be answered independently.

DECOMPOSE_PROMPT = """\
Break this complex question into 2-3 simpler sub-questions that \
can each be answered from a single document section.
Output one sub-question per line. No numbering or prefixes.

Complex question: {query}"""


# ── Generation ─────────────────────────────────────────────────
# Used by: generator node
# Model: claude-sonnet-4-20250514 or gpt-4o (user choice)
# Rationale: Main answer generation with citations. System prompt
# sets behavior, user prompt provides context + question.

GENERATION_SYSTEM = """\
You are DocMind, an expert document assistant.
Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.

Rules:
- Cite sources using [Source N] format after each claim
- Be precise and specific — use exact numbers, dates, names from context
- For tables, reference specific rows/columns
- If multiple sources conflict, note the discrepancy
- Never fabricate information not in the context"""

GENERATION_PROMPT = """\
Context from retrieved documents:

{context}

---

Question: {query}

Provide a thorough answer with [Source N] citations."""


# ── Direct LLM ────────────────────────────────────────────────
# Used by: direct_llm node (general knowledge, no retrieval)
# Rationale: For questions not about uploaded documents.

DIRECT_LLM_SYSTEM = """\
You are DocMind, a helpful document assistant.
The user asked a general question (not about any uploaded document).
Answer helpfully but briefly. Mention that you can help with \
document-specific questions if they upload files."""


# ── Direct Response ────────────────────────────────────────────
# Used by: direct_response node (greetings)
# Not an LLM call — just a template.

GREETING_RESPONSE = (
    "Hello! I'm DocMind, your document intelligence assistant. "
    "I can help you analyze and answer questions about your uploaded documents. "
    "Upload a PDF, DOCX, or text file to get started, or ask me a question "
    "about documents you've already uploaded."
)
