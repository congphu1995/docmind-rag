from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from backend.app.api.dependencies import get_current_user
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.document import Document
from backend.app.vectorstore.factory import VectorStoreFactory

router = APIRouter()


def _build_chunk_tree(all_chunks: list[dict]) -> list[dict]:
    """Build parent-child tree from flat list of ES documents."""
    parents = [c for c in all_chunks if c.get("is_parent")]
    children = [c for c in all_chunks if not c.get("is_parent")]

    children_by_parent: dict[str, list[dict]] = {}
    for child in children:
        pid = child.get("parent_id", "")
        children_by_parent.setdefault(pid, []).append(child)

    tree = []
    for parent in parents:
        tree.append({
            "chunk_id": parent["chunk_id"],
            "content_raw": parent.get("content_raw", ""),
            "content_markdown": parent.get("content_markdown"),
            "content_html": parent.get("content_html"),
            "type": parent.get("type", "text"),
            "page": parent.get("page", 0),
            "section": parent.get("section", ""),
            "language": parent.get("language", "en"),
            "word_count": parent.get("word_count", 0),
            "children": children_by_parent.get(parent["chunk_id"], []),
        })

    # Add orphan children (atomic chunks with no parent)
    parent_ids = {p["chunk_id"] for p in parents}
    orphan_parent_ids = set(children_by_parent.keys()) - parent_ids
    for pid in orphan_parent_ids:
        if not pid:
            for child in children_by_parent[pid]:
                tree.append({
                    "chunk_id": child["chunk_id"],
                    "content_raw": child.get("content_raw", ""),
                    "content_markdown": child.get("content_markdown"),
                    "content_html": child.get("content_html"),
                    "type": child.get("type", "text"),
                    "page": child.get("page", 0),
                    "section": child.get("section", ""),
                    "language": child.get("language", "en"),
                    "word_count": len(child.get("content_raw", "").split()),
                    "children": [],
                })

    return tree


@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    type_filter: str = Query(default=None, description="Filter by chunk type"),
    page_filter: int = Query(default=None, description="Filter by page number"),
    search: str = Query(default=None, description="Search chunk content"),
    user: dict = Depends(get_current_user),
):
    # Verify document ownership
    async with AsyncSessionLocal() as session:
        doc_result = await session.execute(
            select(Document).where(
                Document.doc_id == doc_id,
                Document.user_id == user["user_id"],
            )
        )
        if not doc_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Document not found")

    # Fetch all chunks from vectorstore
    vectorstore = VectorStoreFactory.create()
    all_chunks = await vectorstore.get_by_doc_id(doc_id)

    # Apply filters
    if type_filter:
        all_chunks = [c for c in all_chunks if c.get("type") == type_filter]
    if page_filter is not None:
        all_chunks = [c for c in all_chunks if c.get("page") == page_filter]
    if search:
        search_lower = search.lower()
        all_chunks = [
            c for c in all_chunks
            if search_lower in c.get("content_raw", "").lower()
            or search_lower in c.get("content", "").lower()
        ]

    tree = _build_chunk_tree(all_chunks)
    return {"doc_id": doc_id, "chunks": tree, "total": len(tree)}
