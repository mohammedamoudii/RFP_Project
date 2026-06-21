"""Manual retrieval smoke tests for the RFP and proposal Chroma collections."""

from pathlib import Path
import argparse
import json
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


RFP_DB_PATH = Path("data/chroma/rfp_db")
PROPOSAL_DB_PATH = Path("data/chroma/proposal_db")

RFP_COLLECTION = "rfp_documents"
PROPOSAL_COLLECTION = "proposal_knowledge"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_PATH = Path("data/processed/retrieval_test_results.json")


TARGET_CONFIG = {
    "rfp": {
        "db_path": RFP_DB_PATH,
        "collection_name": RFP_COLLECTION,
        "description": "Current/client RFP documents",
    },
    "proposal": {
        "db_path": PROPOSAL_DB_PATH,
        "collection_name": PROPOSAL_COLLECTION,
        "description": "Old proposals and reusable proposal knowledge",
    },
}


def get_collection(target: str):
    """Open the requested Chroma collection with the shared embedding model."""

    if target not in TARGET_CONFIG:
        raise ValueError(f"Unknown target: {target}")

    config = TARGET_CONFIG[target]

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=str(config["db_path"]))

    collection = client.get_collection(
        name=config["collection_name"],
        embedding_function=embedding_function,
    )

    return collection


def safe_get(metadata: dict, key: str, default: str = ""):
    """Read metadata values while converting missing values to a fallback."""

    value = metadata.get(key, default)

    if value is None:
        return default

    return value


def preview_text(text: str, max_chars: int = 450) -> str:
    """Create a compact single-line preview for terminal output."""

    text = " ".join((text or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def run_query(target: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Run one semantic query against the selected collection."""

    collection = get_collection(target)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    output_rows = []

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        output_rows.append(
            {
                "rank": rank,
                "target": target,
                "query": query,
                "chunk_id": chunk_id,
                "distance": distance,
                "database_target": safe_get(metadata, "database_target"),
                "collection_name": safe_get(metadata, "collection_name"),
                "source_file": safe_get(metadata, "source_file"),
                "file_type": safe_get(metadata, "file_type"),
                "folder_name": safe_get(metadata, "folder_name"),
                "opportunity_id": safe_get(metadata, "opportunity_id"),
                "project_id": safe_get(metadata, "project_id"),
                "project_name": safe_get(metadata, "project_name"),
                "page_numbers": safe_get(metadata, "page_numbers"),
                "page_number_start": safe_get(metadata, "page_number_start"),
                "page_number_end": safe_get(metadata, "page_number_end"),
                "sheet_names": safe_get(metadata, "sheet_names"),
                "sections": safe_get(metadata, "sections"),
                "chunk_word_count": safe_get(metadata, "chunk_word_count"),
                "content_preview": preview_text(document),
                "content": document,
            }
        )

    return output_rows


def print_results(rows: list[dict[str, Any]]):
    """Print ranked retrieval rows in a human-readable format."""

    if not rows:
        print("No results found.")
        return

    target = rows[0]["target"]
    query = rows[0]["query"]

    print("\n" + "=" * 100)
    print(f"TARGET: {target}")
    print(f"QUERY: {query}")
    print("=" * 100)

    for row in rows:
        print(f"\nRank: {row['rank']}")
        print(f"Distance: {row['distance']}")
        print(f"Chunk ID: {row['chunk_id']}")
        print(f"Database target: {row['database_target']}")
        print(f"Collection: {row['collection_name']}")
        print(f"Source file: {row['source_file']}")
        print(f"File type: {row['file_type']}")
        print(f"Folder: {row['folder_name']}")

        if row["page_numbers"]:
            print(f"Pages: {row['page_numbers']}")

        if row["sheet_names"]:
            print(f"Sheets: {row['sheet_names']}")

        if row["sections"]:
            print(f"Sections: {row['sections']}")

        print("\nPreview:")
        print(row["content_preview"])


def save_results(all_rows: list[dict[str, Any]]):
    """Persist retrieval smoke-test results for later inspection."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved retrieval test results to: {OUTPUT_PATH}")


def main():
    """CLI entry point for running the built-in retrieval smoke tests."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=["rfp", "proposal", "both"],
        default="both",
        help="Which database to search.",
    )

    parser.add_argument(
        "--query",
        default=None,
        help="Query text to search.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve.",
    )

    args = parser.parse_args()

    all_rows = []

    if args.target in ["rfp", "both"]:
        rfp_query = args.query or "mandatory requirements submission instructions evaluation criteria technical requirements" # check this one later
        rfp_rows = run_query(
            target="rfp",
            query=rfp_query,
            top_k=args.top_k,
        )
        print_results(rfp_rows)
        all_rows.extend(rfp_rows)

    if args.target in ["proposal", "both"]:
        proposal_query = args.query or "company experience generative AI implementation methodology technical approach"
        proposal_rows = run_query(
            target="proposal",
            query=proposal_query,
            top_k=args.top_k,
        )
        print_results(proposal_rows)
        all_rows.extend(proposal_rows)

    save_results(all_rows)


if __name__ == "__main__":
    main()
