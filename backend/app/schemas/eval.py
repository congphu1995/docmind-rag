from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    dataset: str = Field(
        default="financebench",
        description="Dataset to evaluate against",
    )
    sample_size: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of questions to evaluate",
    )
    config: dict = Field(
        default_factory=dict,
        description="Override chunker/retrieval config for ablation",
    )


class EvalMetrics(BaseModel):
    retrieval_hit_rate: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    latency_p95_ms: float = 0.0
    sample_size: int = 0


class EvalRunResponse(BaseModel):
    run_id: str
    status: str
    dataset: str
    sample_size: int


class EvalResultResponse(BaseModel):
    run_id: str
    status: str
    dataset: str
    metrics: EvalMetrics | None = None
    error: str | None = None
