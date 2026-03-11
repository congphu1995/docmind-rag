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
    content: str
    score: float
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    llm_used: str
    hyde_used: bool
    query_type: str
    agent_trace: list[str]


class QueryFilters(BaseModel):
    """Extracted metadata filters from query."""
    model_config = {"extra": "forbid"}

    doc_type: str = Field(default="", description="Document type filter if mentioned")
    date_range: str = Field(default="", description="Date range if mentioned")
    organization: str = Field(default="", description="Organization filter if mentioned")


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
    filters: QueryFilters = Field(
        default_factory=QueryFilters,
        description="Extracted metadata filters from the query",
    )
