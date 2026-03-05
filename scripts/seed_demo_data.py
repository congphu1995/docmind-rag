#!/usr/bin/env python3
"""
Seed the demo with sample documents.
Usage: uv run python scripts/seed_demo_data.py
Requires: Docker services running (qdrant, postgres, redis)
"""
import asyncio
import sys
from pathlib import Path


async def main():
    # Check for sample docs
    fixture_dir = Path("tests/fixtures")
    if not fixture_dir.exists():
        print("No test fixtures found. Create tests/fixtures/ with sample PDFs.")
        sys.exit(1)

    pdf_files = list(fixture_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files in tests/fixtures/. Add sample PDFs to seed.")
        print("Creating a minimal text document for seeding...")
        sample_path = fixture_dir / "sample_doc.txt"
        sample_path.write_text(
            "DocMind RAG Sample Document\n\n"
            "This is a sample document for testing the DocMind RAG pipeline.\n\n"
            "Section 1: Overview\n"
            "DocMind is a production-grade document intelligence platform.\n"
            "It supports PDF, DOCX, and text files.\n\n"
            "Section 2: Features\n"
            "- Smart document parsing with Docling and PyMuPDF\n"
            "- Parent-child chunking for precise retrieval\n"
            "- Agentic RAG pipeline with LangGraph\n"
            "- Multi-LLM support (Claude + GPT-4o)\n"
            "- Streaming responses with citations\n"
        )
        pdf_files = [sample_path]

    from backend.app.services.ingestion import IngestionService

    service = IngestionService()

    for file_path in pdf_files[:5]:  # Max 5 files
        print(f"Ingesting: {file_path.name}")
        try:
            result = await service.ingest(
                file_path=str(file_path),
                doc_name=file_path.name,
            )
            print(
                f"  Done: {result['elements_parsed']} elements, "
                f"{result['child_chunks']} chunks indexed"
            )
        except Exception as e:
            print(f"  Failed: {e}")

    print("\nSeeding complete.")


if __name__ == "__main__":
    asyncio.run(main())
