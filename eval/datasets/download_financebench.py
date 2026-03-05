#!/usr/bin/env python3
"""
Download FinanceBench dataset from HuggingFace and save as JSON.
Usage: uv run python eval/datasets/download_financebench.py
"""
import json
import sys
from pathlib import Path


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: uv add datasets")
        sys.exit(1)

    print("Downloading FinanceBench from HuggingFace...")
    ds = load_dataset("PatronusAI/financebench", split="train")
    print(f"Loaded {len(ds)} questions")

    output_dir = Path(__file__).parent
    output_path = output_dir / "financebench.json"

    questions = []
    for item in ds:
        questions.append({
            "question": item["question"],
            "answer": item.get("answer", ""),
            "doc_name": item.get("doc_name", ""),
            "page_num": item.get("page_num", ""),
            "category": item.get("question_type", ""),
        })

    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"Saved {len(questions)} questions to {output_path}")

    # Print summary
    categories = {}
    for q in questions:
        cat = q.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
