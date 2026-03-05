from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    llm: str = Field(default="openai", pattern="^(claude|openai)$")
    doc_ids: list[str] = []
    history: list[dict] = []
    stream: bool = True


class ChatSource(BaseModel):
    doc_name: str
    page: int
    section: str
    content_preview: str
    score: float
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    llm_used: str
    hyde_used: bool
    query_type: str
    agent_trace: list[str]


class QueryAnalysis(BaseModel):
    """Structured output for query classification."""
    query_type: str = Field(
        description="One of: factual|analytical|tabular|multi_hop|general|greeting"
    )
    language: str = Field(description="ISO 639-1 code of query language")
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Sub-questions for multi_hop queries, empty otherwise",
    )
    filters: dict = Field(
        default_factory=dict,
        description="Extracted metadata filters: doc_type, date_range, etc.",
    )
