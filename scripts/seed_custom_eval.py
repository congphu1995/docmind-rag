#!/usr/bin/env python3
"""
Seed custom eval documents (ML/AI papers from arXiv).
Downloads PDFs and ingests them via IngestionService.
Usage: uv run python scripts/seed_custom_eval.py
Requires: Docker services running (qdrant, postgres, redis)
"""

import asyncio
import json
import secrets
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EVAL_USER_EMAIL = "eval@docmind.system"
EVAL_USER_NAME = "eval_system"
DATASET_PATH = Path("eval/datasets/custom_dataset.json")
MANIFEST_PATH = Path("eval/datasets/custom_manifest.json")


def load_custom_dataset() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Create the dataset file first."
        )
    return json.loads(DATASET_PATH.read_text())


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


async def download_papers(papers: list[dict], tmp_dir: Path) -> list[dict]:
    import httpx

    docs = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for paper in papers:
            pdf_path = tmp_dir / paper["filename"]
            print(f"  Downloading {paper['title'][:60]}...")
            r = await client.get(paper["pdf_url"])
            if r.status_code != 200 or len(r.content) < 1000:
                raise RuntimeError(
                    f"Download failed for {paper['paper_id']}: "
                    f"status={r.status_code}, size={len(r.content)}"
                )
            pdf_path.write_bytes(r.content)
            docs.append(
                {
                    "paper_id": paper["paper_id"],
                    "filename": paper["filename"],
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
        print(f"Ingesting: {doc['filename']}")
        try:
            result = await service.ingest(
                file_path=doc["path"],
                doc_name=doc["filename"],
                user_id=user_id,
            )
            manifest_docs.append(
                {
                    "paper_id": doc["paper_id"],
                    "filename": doc["filename"],
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
            # Intentional deviation from seed_demo_data.py: fail fast on ingestion
            # errors because every paper must succeed for a curated eval dataset.
            print(f"  Failed: {e}")
            raise

    return manifest_docs


def save_manifest(user_id: str, docs: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "user_id": user_id,
        "documents": docs,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved: {MANIFEST_PATH}")


async def main():
    print("=== DocMind RAG — Seed Custom Eval Papers ===\n")

    dataset = load_custom_dataset()
    papers = dataset["papers"]
    print(f"Dataset v{dataset['version']}: {len(papers)} papers\n")

    user_id = await get_or_create_eval_user()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("\nDownloading papers...")
        docs = await download_papers(papers, tmp_dir)
        print(f"Ready to ingest: {len(docs)} papers\n")

        ingested = await ingest_docs(docs, user_id)

    save_manifest(user_id, ingested)
    print(f"\nSeeding complete. {len(ingested)} papers ingested.")


if __name__ == "__main__":
    asyncio.run(main())
