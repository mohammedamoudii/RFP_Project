from pathlib import Path
import argparse
import json
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


RFP_DB_PATH = Path("data/chroma/rfp_db")
RFP_COLLECTION = "rfp_documents"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_PATH = Path("data/processed/retrieved_rfp_context.json")


DEFAULT_REQUIREMENT_QUERIES = [
    "mandatory requirements submission instructions compliance requirements",
    "technical requirements scope of work deliverables implementation requirements",
    "evaluation criteria scoring criteria proposal response requirements",
    "security privacy data governance requirements",
    "timeline deadline milestones project schedule requirements",
]


def get_collection():
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=str(RFP_DB_PATH))

    collection = client.get_collection(
        name=RFP_COLLECTION,
        embedding_function=embedding_function,
    )

    return collection


def preview_text(text: str, max_chars: int = 400) -> str:
    text = " ".join((text or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def safe_get(metadata: dict, key: str, default: str = ""):
    value = metadata.get(key, default)

    if value is None:
        return default

    return value


def query_rfp_opportunity(
    opportunity_id: str,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"opportunity_id": opportunity_id},
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    rows = []

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "query": query,
                "chunk_id": chunk_id,
                "distance": distance,
                "database_target": safe_get(metadata, "database_target"),
                "collection_name": safe_get(metadata, "collection_name"),
                "opportunity_id": safe_get(metadata, "opportunity_id"),
                "folder_name": safe_get(metadata, "folder_name"),
                "source_file": safe_get(metadata, "source_file"),
                "file_type": safe_get(metadata, "file_type"),
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

    return rows


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []

    for row in rows:
        chunk_id = row["chunk_id"]

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        deduped.append(row)

    return deduped


def print_results(rows: list[dict[str, Any]], max_preview: int = 10):
    print(f"\nRetrieved unique chunks: {len(rows)}")

    for row in rows[:max_preview]:
        print("\n" + "-" * 100)
        print(f"Rank: {row['rank']}")
        print(f"Query: {row['query']}")
        print(f"Chunk ID: {row['chunk_id']}")
        print(f"Distance: {row['distance']}")
        print(f"Opportunity ID: {row['opportunity_id']}")
        print(f"Database target: {row['database_target']}")
        print(f"Collection: {row['collection_name']}")
        print(f"Source file: {row['source_file']}")
        print(f"File type: {row['file_type']}")

        if row["page_numbers"]:
            print(f"Pages: {row['page_numbers']}")

        if row["sheet_names"]:
            print(f"Sheets: {row['sheet_names']}")

        print("\nPreview:")
        print(row["content_preview"])


def save_results(rows: list[dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "retrieval_type": "rfp_requirement_context",
        "total_chunks": len(rows),
        "chunks": rows,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved RFP context to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--opportunity-id",
        required=True,
        help="The opportunity_id to retrieve from.",
    )

    parser.add_argument(
        "--query",
        default=None,
        help="Optional single query. If not provided, default requirement queries are used.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-k results per query.",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output JSON path.",
    )

    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_REQUIREMENT_QUERIES

    all_rows = []

    for query in queries:
        rows = query_rfp_opportunity(
            opportunity_id=args.opportunity_id,
            query=query,
            top_k=args.top_k,
        )

        all_rows.extend(rows)

    deduped_rows = deduplicate_rows(all_rows)

    print_results(deduped_rows)
    save_results(deduped_rows, Path(args.output))


if __name__ == "__main__":
    main()