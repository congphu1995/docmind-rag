from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from backend.app.api.dependencies import get_current_user
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.document import Document, ParentChunk
from backend.app.vectorstore.qdrant_client import QdrantWrapper

router = APIRouter()


def _build_chunk_tree(parents: list, children: list[dict]) -> list[dict]:
    """Build parent-child tree from flat lists."""
    tree = []
    children_by_parent = {}
    for child in children:
        pid = child.get("parent_id", "")
        children_by_parent.setdefault(pid, []).append(child)

    for parent in parents:
        tree.append({
            "chunk_id": parent.chunk_id,
            "content_raw": parent.content_raw,
            "content_markdown": parent.content_markdown,
            "content_html": parent.content_html,
            "type": parent.type,
            "page": parent.page,
            "section": parent.section,
            "language": parent.language,
            "word_count": parent.word_count,
            "children": children_by_parent.get(parent.chunk_id, []),
        })

    # Add orphan children (atomic chunks with no parent)
    orphan_parent_ids = set(children_by_parent.keys()) - {p.chunk_id for p in parents}
    for pid in orphan_parent_ids:
        if not pid:  # empty parent_id = atomic
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

        # Fetch parents from PostgreSQL
        query = select(ParentChunk).where(ParentChunk.doc_id == doc_id)
        if type_filter:
            query = query.where(ParentChunk.type == type_filter)
        if page_filter is not None:
            query = query.where(ParentChunk.page == page_filter)

        result = await session.execute(query)
        parents = result.scalars().all()

    # Fetch children from Qdrant
    qdrant = QdrantWrapper()
    children = await qdrant.get_by_doc_id(doc_id)

    # Apply filters to children
    if type_filter:
        children = [c for c in children if c.get("type") == type_filter]
    if page_filter is not None:
        children = [c for c in children if c.get("page") == page_filter]
    if search:
        search_lower = search.lower()
        children = [
            c for c in children
            if search_lower in c.get("content_raw", "").lower()
            or search_lower in c.get("content", "").lower()
        ]
        parents = [
            p for p in parents
            if search_lower in (p.content_raw or "").lower()
        ]

    tree = _build_chunk_tree(parents, children)
    return {"doc_id": doc_id, "chunks": tree, "total": len(tree)}
