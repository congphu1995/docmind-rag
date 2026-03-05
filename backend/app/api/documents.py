import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.core.logging import logger
from backend.app.workers.ingest_tasks import ingest_document_task

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
    parser_strategy: str = Form(default="auto"),
):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size: 50MB",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(
        "upload_received",
        filename=file.filename,
        size_mb=round(len(content) / 1024 / 1024, 2),
    )

    task = ingest_document_task.delay(
        file_path=tmp_path,
        doc_name=file.filename,
        language=language,
        parser_strategy=parser_strategy,
    )

    return JSONResponse(
        {
            "status": "processing",
            "task_id": task.id,
            "message": f"Document '{file.filename}' queued for processing",
        }
    )


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = ingest_document_task.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"status": "ready", "result": task.result}
    elif task.state == "FAILURE":
        return {"status": "failed", "error": str(task.result)}
    else:
        return {"status": task.state.lower()}


@router.get("/")
async def list_documents():
    from sqlalchemy import select

    from backend.app.core.database import AsyncSessionLocal
    from backend.app.models.document import Document

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document))
        docs = result.scalars().all()
    return {
        "documents": [
            {"doc_id": d.doc_id, "doc_name": d.doc_name, "status": d.status}
            for d in docs
        ],
        "total": len(docs),
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    from backend.app.services.ingestion import IngestionService

    service = IngestionService()
    await service.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
