#!/usr/bin/env python3
"""
Seed FinanceBench documents for evaluation.
Creates a system eval user, downloads 5 10-K PDFs, ingests them.
Usage: uv run python scripts/seed_demo_data.py
Requires: Docker services running (elasticsearch, postgres, redis)
"""

import asyncio
import json
import secrets
import tempfile
from pathlib import Path

EVAL_USER_EMAIL = "eval@docmind.system"
EVAL_USER_NAME = "eval_system"
MANIFEST_PATH = Path("eval/datasets/seed_manifest.json")

# Must match exact doc_name values from PatronusAI/financebench dataset
FINANCEBENCH_DOCS = [
    "MICROSOFT_2023_10K",
    "AMAZON_2019_10K",
    "COCACOLA_2022_10K",
    "NIKE_2023_10K",
    "PEPSICO_2022_10K",
]


async def get_or_create_eval_user() -> str:
    from backend.app.core.database import AsyncSessionLocal, create_tables
    from backend.app.models.user import User
    from sqlalchemy import select

    await create_tables()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.email == EVAL_USER_EMAIL)
        )
        user = result.scalar_one_or_none()
        if user:
            print(f"Eval user exists: {user.id}")
            return user.id

    from backend.app.services.auth import AuthService

    auth = AuthService()
    user = await auth.register(
        email=EVAL_USER_EMAIL,
        username=EVAL_USER_NAME,
        password=secrets.token_urlsafe(32),
    )
    print(f"Created eval user: {user.id}")
    return user.id


async def download_financebench_pdfs(tmp_dir: Path) -> list[dict]:
    import httpx
    from datasets import load_dataset

    ds = load_dataset("PatronusAI/financebench", split="train")

    # Collect unique doc_link per target doc_name
    targets = set(FINANCEBENCH_DOCS)
    links: dict[str, str] = {}
    for item in ds:
        doc_name = item.get("doc_name", "")
        doc_link = item.get("doc_link", "")
        if doc_name in targets and doc_link and doc_name not in links:
            links[doc_name] = doc_link
        if len(links) == len(targets):
            break

    missing = targets - set(links.keys())
    if missing:
        raise RuntimeError(f"Could not find doc_links for: {missing}")

    docs = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for doc_name, doc_link in links.items():
            pdf_path = tmp_dir / f"{doc_name}.pdf"
            print(f"  Downloading {doc_name}...")
            r = await client.get(doc_link)
            if r.status_code != 200 or len(r.content) < 1000:
                raise RuntimeError(
                    f"Download failed for {doc_name}: "
                    f"status={r.status_code}, size={len(r.content)}"
                )
            pdf_path.write_bytes(r.content)
            docs.append(
                {
                    "doc_name": f"{doc_name}.pdf",
                    "hf_doc_name": doc_name,
                    "path": str(pdf_path),
                }
            )
            print(f"    OK ({len(r.content) // 1024} KB)")

    return docs


async def ingest_docs(docs: list[dict], user_id: str) -> list[dict]:
    from backend.app.services.ingestion import IngestionService

    service = IngestionService()

    manifest_docs = []
    for doc in docs:
        print(f"Ingesting: {doc['doc_name']}")
        try:
            result = await service.ingest(
                file_path=doc["path"],
                doc_name=doc["doc_name"],
                user_id=user_id,
            )
            manifest_docs.append(
                {
                    "doc_name": doc["doc_name"],
                    "hf_doc_name": doc["hf_doc_name"],
                    "doc_id": result["doc_id"],
                    "elements_parsed": result["elements_parsed"],
                    "child_chunks": result["child_chunks"],
                }
            )
            print(
                f"  Done: {result['elements_parsed']} elements, "
                f"{result['child_chunks']} chunks indexed"
            )
        except Exception as e:
            print(f"  Failed: {e}")

    return manifest_docs


async def save_manifest(user_id: str, docs: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "user_id": user_id,
        "documents": docs,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved: {MANIFEST_PATH}")


async def main():
    print("=== DocMind RAG — Seed FinanceBench ===\n")

    user_id = await get_or_create_eval_user()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("\nDownloading FinanceBench documents...")
        docs = await download_financebench_pdfs(tmp_dir)
        print(f"Ready to ingest: {len(docs)} documents\n")

        ingested = await ingest_docs(docs, user_id)

    await save_manifest(user_id, ingested)
    print(f"\nSeeding complete. {len(ingested)} documents ingested.")


if __name__ == "__main__":
    asyncio.run(main())
