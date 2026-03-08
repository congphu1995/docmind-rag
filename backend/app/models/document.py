from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, Text
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


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    chunk_id = Column(String, primary_key=True)
    doc_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=True, index=True)
    content_raw = Column(Text, nullable=False)
    content_markdown = Column(Text)
    content_html = Column(Text)
    type = Column(String, default="text")
    page = Column(Integer, default=0)
    section = Column(String, default="")
    language = Column(String, default="en")
    word_count = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
