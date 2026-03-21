from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
    doc_name = Column(String, nullable=False)
    user_id = Column(String, nullable=True, index=True)
    file_path = Column(String)
    language = Column(String, default="en")
    doc_type = Column(String, default="document")
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    parser_used = Column(String)
    status = Column(String, default="processing")
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
