from pathlib import Path
import argparse
import json
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


EXTRACTED_REQUIREMENTS_PATH = Path("data/processed/extracted_requirements.json")
OUTPUT_PATH = Path("data/processed/retrieved_proposal_context.json")

PROPOSAL_DB_PATH = Path("data/chroma/proposal_db")
PROPOSAL_COLLECTION = "proposal_knowledge"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


BUSINESS_CATEGORIES = {
    "business",
    "delivery",
    "pricing",
    "evaluation",
    "submission",
    "legal",
    "other",
}

TECHNICAL_CATEGORIES = {
    "technical",
    "security",
    "compliance",
    "delivery",
    "other",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_collection():
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=str(PROPOSAL_DB_PATH))

    collection = client.get_collection(
        name=PROPOSAL_COLLECTION,
        embedding_function=embedding_function,
    )

    return collection


def safe_get(metadata: dict, key: str, default: str = ""):
    value = metadata.get(key, default)

    if value is None:
        return default

    return value


def preview_text(text: str, max_chars: int = 450) -> str:
    text = " ".join((text or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def requirement_matches_proposal_type(requirement: dict[str, Any], proposal_type: str) -> bool:
    if proposal_type == "both":
        return True

    req_proposal_types = requirement.get("proposal_type", [])
    category = requirement.get("category", "other")

    if proposal_type == "business":
        return "business" in req_proposal_types or category in BUSINESS_CATEGORIES

    if proposal_type == "technical":
        return "technical" in req_proposal_types or category in TECHNICAL_CATEGORIES

    return True


def select_requirements(requirements: list[dict[str, Any]], proposal_type: str) -> list[dict[str, Any]]:
    selected = [
        req for req in requirements
        if requirement_matches_proposal_type(req, proposal_type)
    ]

    # Always prefer mandatory requirements first.
    selected = sorted(
        selected,
        key=lambda req: (
            0 if req.get("priority") == "mandatory" else 1,
            req.get("requirement_id", ""),
        ),
    )

    return selected


def build_query_from_requirement(requirement: dict[str, Any], proposal_type: str) -> str:
    requirement_text = requirement.get("requirement_text", "")
    category = requirement.get("category", "")
    priority = requirement.get("priority", "")
    evidence = requirement.get("evidence_quote", "")

    if proposal_type == "technical":
        intent = "technical approach implementation methodology security architecture solution delivery"
    elif proposal_type == "business":
        intent = "company experience business value delivery plan methodology project management response"
    else:
        intent = "company experience methodology technical approach business value implementation delivery"

    query = f"""
{intent}

Requirement:
{requirement_text}

Category: {category}
Priority: {priority}
Evidence:
{evidence}
""".strip()

    return query


def add_general_queries(proposal_type: str) -> list[dict[str, str]]:
    if proposal_type == "technical":
        return [
            {
                "query": "technical approach implementation methodology solution architecture integration security",
                "query_type": "general_technical",
                "requirement_id": "GENERAL-TECHNICAL",
            },
            {
                "query": "AI software implementation data privacy security governance testing deployment",
                "query_type": "general_technical",
                "requirement_id": "GENERAL-TECHNICAL",
            },
        ]

    if proposal_type == "business":
        return [
            {
                "query": "company experience project management delivery plan business value implementation timeline",
                "query_type": "general_business",
                "requirement_id": "GENERAL-BUSINESS",
            },
            {
                "query": "proposal response executive summary relevant experience client outcomes methodology",
                "query_type": "general_business",
                "requirement_id": "GENERAL-BUSINESS",
            },
        ]

    return [
        {
            "query": "company experience methodology technical approach implementation business value delivery plan",
            "query_type": "general_both",
            "requirement_id": "GENERAL-BOTH",
        },
        {
            "query": "AI software implementation project management security data governance solution delivery",
            "query_type": "general_both",
            "requirement_id": "GENERAL-BOTH",
        },
    ]


def build_queries(
    selected_requirements: list[dict[str, Any]],
    proposal_type: str,
    max_requirement_queries: int,
) -> list[dict[str, str]]:
    queries = []

    for req in selected_requirements[:max_requirement_queries]:
        queries.append(
            {
                "query": build_query_from_requirement(req, proposal_type),
                "query_type": "requirement",
                "requirement_id": req.get("requirement_id", ""),
            }
        )

    queries.extend(add_general_queries(proposal_type))

    return queries


def query_proposal_db(query: str, top_k: int) -> list[dict[str, Any]]:
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
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
                "chunk_id": chunk_id,
                "distance": distance,
                "database_target": safe_get(metadata, "database_target"),
                "collection_name": safe_get(metadata, "collection_name"),
                "source_file": safe_get(metadata, "source_file"),
                "file_type": safe_get(metadata, "file_type"),
                "folder_name": safe_get(metadata, "folder_name"),
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

    return rows


def deduplicate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_chunk = {}

    for row in rows:
        chunk_id = row["chunk_id"]

        if chunk_id not in best_by_chunk:
            best_by_chunk[chunk_id] = row
            continue

        old_distance = best_by_chunk[chunk_id]["distance"]
        new_distance = row["distance"]

        if new_distance < old_distance:
            best_by_chunk[chunk_id] = row

    deduped = list(best_by_chunk.values())
    deduped = sorted(deduped, key=lambda row: row["distance"])

    return deduped


def print_results(rows: list[dict[str, Any]], max_preview: int = 10):
    print(f"\nRetrieved unique proposal chunks: {len(rows)}")

    for row in rows[:max_preview]:
        print("\n" + "-" * 100)
        print(f"Rank: {row['rank']}")
        print(f"Matched requirement: {row.get('matched_requirement_id')}")
        print(f"Query type: {row.get('query_type')}")
        print(f"Distance: {row['distance']}")
        print(f"Chunk ID: {row['chunk_id']}")
        print(f"Database target: {row['database_target']}")
        print(f"Collection: {row['collection_name']}")
        print(f"Source file: {row['source_file']}")
        print(f"Project name: {row['project_name']}")

        if row["page_numbers"]:
            print(f"Pages: {row['page_numbers']}")

        print("\nPreview:")
        print(row["content_preview"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--requirements",
        default=str(EXTRACTED_REQUIREMENTS_PATH),
        help="Path to extracted requirements JSON.",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to save retrieved proposal context.",
    )

    parser.add_argument(
        "--proposal-type",
        choices=["business", "technical", "both"],
        default="both",
        help="Proposal type to retrieve for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k chunks per query.",
    )

    parser.add_argument(
        "--max-requirement-queries",
        type=int,
        default=12,
        help="Maximum number of requirement-based queries.",
    )

    args = parser.parse_args()

    extracted = load_json(Path(args.requirements))
    requirements = extracted.get("requirements", [])

    if not requirements:
        raise ValueError("No requirements found. Run extract_requirements.py first.")

    selected_requirements = select_requirements(
        requirements=requirements,
        proposal_type=args.proposal_type,
    )

    queries = build_queries(
        selected_requirements=selected_requirements,
        proposal_type=args.proposal_type,
        max_requirement_queries=args.max_requirement_queries,
    )

    all_rows = []

    print(f"Opportunity ID: {extracted.get('opportunity_id')}")
    print(f"Proposal type: {args.proposal_type}")
    print(f"Total requirements: {len(requirements)}")
    print(f"Selected requirements: {len(selected_requirements)}")
    print(f"Queries: {len(queries)}")
    print(f"Top-k per query: {args.top_k}")

    for query_item in queries:
        query = query_item["query"]
        rows = query_proposal_db(
            query=query,
            top_k=args.top_k,
        )

        for row in rows:
            row["matched_requirement_id"] = query_item["requirement_id"]
            row["query_type"] = query_item["query_type"]
            row["query"] = query

        all_rows.extend(rows)

    deduped_rows = deduplicate_results(all_rows)

    print_results(deduped_rows)

    payload = {
        "retrieval_type": "proposal_knowledge_context",
        "opportunity_id": extracted.get("opportunity_id"),
        "proposal_type": args.proposal_type,
        "total_requirements": len(requirements),
        "selected_requirements_count": len(selected_requirements),
        "queries_count": len(queries),
        "top_k_per_query": args.top_k,
        "selected_requirements": selected_requirements,
        "queries": queries,
        "total_retrieved_chunks_before_dedup": len(all_rows),
        "total_retrieved_chunks_after_dedup": len(deduped_rows),
        "chunks": deduped_rows,
    }

    save_json(payload, Path(args.output))

    print(f"\nSaved retrieved proposal context to: {args.output}")


if __name__ == "__main__":
    main()