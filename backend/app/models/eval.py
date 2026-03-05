from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String

from backend.app.models.document import Base


class EvalRun(Base):
    __tablename__ = "eval_runs"

    run_id = Column(String, primary_key=True)
    dataset = Column(String, nullable=False)
    sample_size = Column(Integer, default=0)
    status = Column(String, default="pending")
    config = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    error = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
