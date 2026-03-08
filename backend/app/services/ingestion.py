"""
Orchestrates the full ingestion pipeline:
preprocess → parse → normalize → extract_metadata → chunk → enrich → filter → embed → store
"""
import os
import time
import uuid

from backend.app.core.database import AsyncSessionLocal
from backend.app.core.exceptions import IngestionError
from backend.app.core.logging import logger
from backend.app.core.metrics import (
    CHUNKS_CREATED_TOTAL,
    INGESTION_DOCUMENTS_TOTAL,
    INGESTION_STAGE_DURATION,
)
from backend.app.models.document import Document, ParentChunk
from backend.app.pipeline.chunkers.enricher import ContextEnricher
from backend.app.pipeline.chunkers.quality_filter import QualityFilter
from backend.app.pipeline.chunkers.smart_router import SmartRouter
from backend.app.pipeline.embedders.openai_embedder import OpenAIEmbedder
from backend.app.pipeline.llm.factory import LLMFactory
from backend.app.pipeline.parsers.factory import ParserFactory
from backend.app.pipeline.parsers.metadata_extractor import MetadataExtractor
from backend.app.pipeline.multimodal.figure_describer import FigureDescriber
from backend.app.pipeline.multimodal.table_representer import TableRepresenter
from backend.app.pipeline.parsers.preprocessor import PDFPreprocessor
from backend.app.vectorstore.qdrant_client import QdrantWrapper


class IngestionService:

    def __init__(self):
        mini_llm = LLMFactory.create_mini()
        vision_llm = LLMFactory.create("openai", model="gpt-4o")
        self._preprocessor = PDFPreprocessor()
        self._metadata_extractor = MetadataExtractor(llm=mini_llm)
        self._figure_describer = FigureDescriber(llm=vision_llm)
        self._table_representer = TableRepresenter(llm=mini_llm)
        self._router = SmartRouter(
            figure_describer=self._figure_describer,
            table_representer=self._table_representer,
        )
        self._enricher = ContextEnricher(llm=mini_llm)
        self._quality_filter = QualityFilter()
        self._embedder = OpenAIEmbedder()
        self._qdrant = QdrantWrapper()

    async def ingest(
        self,
        file_path: str,
        doc_name: str,
        language: str = "en",
        parser_strategy: str = "auto",
        user_id: str = None,
    ) -> dict:
        doc_id = str(uuid.uuid4())[:12]
        log = logger.bind(doc_id=doc_id, doc_name=doc_name)
        log.info("ingestion_start")

        try:
            # 1. Pre-process
            log.info("stage_preprocess")
            start = time.perf_counter()
            clean_path = self._preprocessor.preprocess(file_path)
            INGESTION_STAGE_DURATION.labels(stage="preprocess").observe(
                time.perf_counter() - start
            )

            # 2. Parse
            log.info("stage_parse", strategy=parser_strategy)
            start = time.perf_counter()
            if parser_strategy == "auto":
                parser = ParserFactory.auto_select(clean_path)
            else:
                parser = ParserFactory.create(parser_strategy)

            elements = await parser.parse(
                clean_path, doc_id=doc_id, doc_name=doc_name
            )
            INGESTION_STAGE_DURATION.labels(stage="parse").observe(
                time.perf_counter() - start
            )
            log.info(
                "stage_parse_done",
                elements=len(elements),
                parser=type(parser).__name__,
            )

            if not elements:
                raise IngestionError("No content extracted from document")

            detected_lang = next(
                (el.language for el in elements if el.language), language
            )

            # 3. Extract doc-level metadata
            log.info("stage_metadata")
            start = time.perf_counter()
            doc_metadata = await self._metadata_extractor.extract(elements)
            doc_metadata.update(
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "language": detected_lang,
                    "parser": type(parser).__name__,
                }
            )
            INGESTION_STAGE_DURATION.labels(stage="metadata").observe(
                time.perf_counter() - start
            )

            # 4. Chunk
            log.info("stage_chunk")
            start = time.perf_counter()
            parent_chunks, child_chunks = await self._router.route(
                elements, doc_metadata
            )
            INGESTION_STAGE_DURATION.labels(stage="chunk").observe(
                time.perf_counter() - start
            )
            log.info(
                "stage_chunk_done",
                parents=len(parent_chunks),
                children=len(child_chunks),
            )

            # 5. Enrich child chunks
            log.info("stage_enrich")
            start = time.perf_counter()
            enriched_chunks = await self._enricher.enrich_batch(
                [c for c in child_chunks if not c.is_parent],
                doc_metadata,
            )

            # 6. Quality filter
            final_children = self._quality_filter.filter(enriched_chunks)
            INGESTION_STAGE_DURATION.labels(stage="enrich").observe(
                time.perf_counter() - start
            )
            log.info("stage_enrich_done", after_filter=len(final_children))

            # 7. Embed children
            log.info("stage_embed")
            start = time.perf_counter()
            texts_to_embed = [c.content for c in final_children]
            vectors = await self._embedder.embed(texts_to_embed)
            INGESTION_STAGE_DURATION.labels(stage="embed").observe(
                time.perf_counter() - start
            )

            # 8. Store in Qdrant (children)
            log.info("stage_store_qdrant")
            start = time.perf_counter()
            await self._qdrant.upsert(final_children, vectors)
            INGESTION_STAGE_DURATION.labels(stage="store_qdrant").observe(
                time.perf_counter() - start
            )

            # 9. Store parents in PostgreSQL
            log.info("stage_store_postgres")
            start = time.perf_counter()
            await self._store_parents(
                doc_id, doc_name, parent_chunks, doc_metadata, user_id=user_id
            )
            INGESTION_STAGE_DURATION.labels(stage="store_postgres").observe(
                time.perf_counter() - start
            )

            INGESTION_DOCUMENTS_TOTAL.labels(status="success").inc()
            CHUNKS_CREATED_TOTAL.labels(type="parent").inc(len(parent_chunks))
            CHUNKS_CREATED_TOTAL.labels(type="child").inc(len(final_children))

            result = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "language": detected_lang,
                "parser_used": type(parser).__name__,
                "elements_parsed": len(elements),
                "parent_chunks": len(parent_chunks),
                "child_chunks": len(final_children),
                "doc_type": doc_metadata.get("doc_type", "document"),
            }
            log.info("ingestion_done", **result)
            return result

        except Exception as e:
            log.error("ingestion_failed", error=str(e))
            INGESTION_DOCUMENTS_TOTAL.labels(status="fail").inc()
            raise IngestionError(f"Ingestion failed for {doc_name}: {e}") from e
        finally:
            if clean_path != file_path and os.path.exists(clean_path):
                os.unlink(clean_path)

    async def _store_parents(
        self,
        doc_id: str,
        doc_name: str,
        parent_chunks: list,
        doc_metadata: dict,
        user_id: str = None,
    ):
        async with AsyncSessionLocal() as session:
            # Store document record
            doc = Document(
                doc_id=doc_id,
                doc_name=doc_name,
                user_id=user_id,
                language=doc_metadata.get("language", "en"),
                doc_type=doc_metadata.get("doc_type", "document"),
                chunk_count=len(parent_chunks),
                parser_used=doc_metadata.get("parser", ""),
                status="ready",
                metadata_=doc_metadata,
            )
            session.add(doc)

            for chunk in parent_chunks:
                pg_chunk = ParentChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=doc_id,
                    user_id=user_id,
                    content_raw=chunk.content_raw,
                    content_markdown=chunk.content_markdown,
                    content_html=chunk.content_html,
                    type=chunk.type,
                    page=chunk.page,
                    section=chunk.section,
                    language=chunk.language,
                    word_count=chunk.word_count,
                    metadata_=doc_metadata,
                )
                session.add(pg_chunk)
            await session.commit()

    async def delete_document(self, doc_id: str):
        await self._qdrant.delete_by_doc_id(doc_id)
        async with AsyncSessionLocal() as session:
            from sqlalchemy import delete

            await session.execute(
                delete(ParentChunk).where(ParentChunk.doc_id == doc_id)
            )
            await session.execute(
                delete(Document).where(Document.doc_id == doc_id)
            )
            await session.commit()
