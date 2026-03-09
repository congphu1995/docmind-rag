#!/usr/bin/env python3
"""
Seed FinanceBench documents for evaluation.
Creates a system eval user, downloads 5 10-K PDFs, ingests them.
Usage: uv run python scripts/seed_demo_data.py
Requires: Docker services running (qdrant, postgres, redis)
"""
import asyncio
import json
import secrets
import tempfile
from pathlib import Path

EVAL_USER_EMAIL = "eval@docmind.system"
EVAL_USER_NAME = "eval_system"
MANIFEST_PATH = Path("eval/datasets/seed_manifest.json")

FINANCEBENCH_DOCS = [
    {"doc_name": "AAPL_10K_2023.pdf", "hf_doc_name": "APPLE INC"},
    {"doc_name": "MSFT_10K_2023.pdf", "hf_doc_name": "MICROSOFT CORP"},
    {"doc_name": "AMZN_10K_2023.pdf", "hf_doc_name": "AMAZON COM INC"},
    {"doc_name": "GOOG_10K_2023.pdf", "hf_doc_name": "ALPHABET INC"},
    {"doc_name": "META_10K_2023.pdf", "hf_doc_name": "META PLATFORMS INC"},
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
    docs = []
    try:
        from datasets import load_dataset

        ds = load_dataset("PatronusAI/financebench", split="train")

        seen = set()
        for item in ds:
            doc_name = item.get("doc_name", "")
            doc_link = item.get("doc_link", "")
            if doc_link and doc_name not in seen:
                for target in FINANCEBENCH_DOCS:
                    if target["hf_doc_name"].lower() in doc_name.lower():
                        seen.add(doc_name)
                        pdf_path = tmp_dir / target["doc_name"]
                        if not pdf_path.exists():
                            import httpx
                            print(f"  Downloading {target['doc_name']}...")
                            try:
                                async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                                    r = await client.get(doc_link)
                                    if r.status_code == 200 and len(r.content) > 1000:
                                        pdf_path.write_bytes(r.content)
                                        docs.append({
                                            "doc_name": target["doc_name"],
                                            "hf_doc_name": doc_name,
                                            "path": str(pdf_path),
                                        })
                                    else:
                                        print(f"    Skip: status={r.status_code}, size={len(r.content)}")
                            except Exception as e:
                                print(f"    Download failed: {e}")
                        break
            if len(docs) >= 5:
                break
    except Exception as e:
        print(f"FinanceBench download failed: {e}")

    if not docs:
        print("Creating synthetic seed documents as fallback...")
        docs = create_synthetic_docs(tmp_dir)

    return docs


def create_synthetic_docs(tmp_dir: Path) -> list[dict]:
    docs = []
    for i, company in enumerate(["Acme Corp", "Beta Inc", "Gamma LLC", "Delta Co", "Echo Ltd"]):
        path = tmp_dir / f"synthetic_{i+1}.txt"
        path.write_text(
            f"{company} Annual Report 2023\n\n"
            f"Financial Overview\n"
            f"Total revenue for fiscal year 2023 was ${(i+1)*10}B.\n"
            f"Operating income was ${(i+1)*2}B.\n\n"
            f"Business Segments\n"
            f"{company} operates in {i+2} segments globally.\n"
            f"The largest segment contributed {50+i*5}% of revenue.\n\n"
            f"Risk Factors\n"
            f"Key risks include market competition and regulatory changes.\n"
        )
        docs.append({"doc_name": path.name, "path": str(path), "hf_doc_name": company})
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
            manifest_docs.append({
                "doc_name": doc["doc_name"],
                "hf_doc_name": doc["hf_doc_name"],
                "doc_id": result["doc_id"],
                "elements_parsed": result["elements_parsed"],
                "child_chunks": result["child_chunks"],
            })
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
