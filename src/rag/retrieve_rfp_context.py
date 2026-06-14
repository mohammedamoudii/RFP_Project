from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any, Iterable

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


RFP_DB_PATH = Path("data/chroma/rfp_db")
RFP_COLLECTION = "rfp_documents"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

OUTPUT_PATH = Path(
    "data/processed/retrieved_rfp_context.json"
)

DEFAULT_REQUIREMENT_QUERIES = [
    (
        "mandatory requirements submission instructions "
        "compliance requirements"
    ),
    (
        "technical requirements scope of work deliverables "
        "implementation requirements"
    ),
    (
        "evaluation criteria scoring criteria proposal "
        "response requirements"
    ),
    (
        "security privacy data governance requirements"
    ),
    (
        "timeline deadline milestones project schedule "
        "requirements"
    ),
]


def create_embedding_function(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def get_collection(
    embedding_function=None,
):
    if embedding_function is None:
        embedding_function = (
            create_embedding_function()
        )

    client = chromadb.PersistentClient(
        path=str(RFP_DB_PATH)
    )

    return client.get_collection(
        name=RFP_COLLECTION,
        embedding_function=embedding_function,
    )


def preview_text(
    text: str,
    max_chars: int = 400,
) -> str:
    normalized = " ".join(
        (text or "").split()
    )

    if len(normalized) <= max_chars:
        return normalized

    return (
        normalized[:max_chars]
        + "..."
    )


def safe_get(
    metadata: dict[str, Any],
    key: str,
    default: Any = "",
) -> Any:
    value = metadata.get(
        key,
        default,
    )

    if value is None:
        return default

    return value


def validate_retrieval_inputs(
    query: str,
    opportunity_id: str,
    top_k: int,
) -> tuple[str, str, int]:
    normalized_query = str(
        query
    ).strip()

    normalized_opportunity_id = str(
        opportunity_id
    ).strip()

    if not normalized_query:
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    if not normalized_opportunity_id:
        raise ValueError(
            "Opportunity ID cannot be empty."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    return (
        normalized_query,
        normalized_opportunity_id,
        top_k,
    )


def retrieve_rfp_chunks(
    query: str,
    opportunity_id: str,
    top_k: int = 5,
    collection=None,
) -> list[dict[str, Any]]:
    """
    Retrieve semantically similar chunks only from the
    active RFP opportunity.

    This is the reusable interface for Streamlit,
    requirement extraction, tests, and other modules.
    """
    (
        normalized_query,
        normalized_opportunity_id,
        top_k,
    ) = validate_retrieval_inputs(
        query=query,
        opportunity_id=opportunity_id,
        top_k=top_k,
    )

    if collection is None:
        collection = get_collection()

    results = collection.query(
        query_texts=[
            normalized_query
        ],
        n_results=top_k,
        where={
            "opportunity_id": (
                normalized_opportunity_id
            )
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = (
        results.get(
            "ids",
            [[]],
        )[0]
        or []
    )

    documents = (
        results.get(
            "documents",
            [[]],
        )[0]
        or []
    )

    metadatas = (
        results.get(
            "metadatas",
            [[]],
        )[0]
        or []
    )

    distances = (
        results.get(
            "distances",
            [[]],
        )[0]
        or []
    )

    result_lengths = {
        len(ids),
        len(documents),
        len(metadatas),
        len(distances),
    }

    if len(result_lengths) != 1:
        raise RuntimeError(
            "Chroma returned mismatched retrieval arrays."
        )

    rows: list[dict[str, Any]] = []

    for rank, (
        chunk_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        metadata = metadata or {}

        returned_opportunity = str(
            safe_get(
                metadata,
                "opportunity_id",
                "",
            )
        ).strip()

        returned_target = str(
            safe_get(
                metadata,
                "database_target",
                "",
            )
        ).strip()

        returned_collection = str(
            safe_get(
                metadata,
                "collection_name",
                "",
            )
        ).strip()

        if (
            returned_opportunity
            != normalized_opportunity_id
        ):
            raise RuntimeError(
                "Retrieval scope violation: "
                "a chunk from another opportunity "
                "was returned."
            )

        if returned_target != "rfp_db":
            raise RuntimeError(
                "Retrieval scope violation: "
                "a non-RFP database chunk was returned."
            )

        if (
            returned_collection
            != RFP_COLLECTION
        ):
            raise RuntimeError(
                "Retrieval scope violation: "
                "a chunk from the wrong collection "
                "was returned."
            )

        rows.append(
            {
                "rank": rank,
                "query": normalized_query,
                "chunk_id": chunk_id,
                "distance": distance,
                "database_target": (
                    returned_target
                ),
                "collection_name": (
                    returned_collection
                ),
                "opportunity_id": (
                    returned_opportunity
                ),
                "folder_name": safe_get(
                    metadata,
                    "folder_name",
                ),
                "source_file": safe_get(
                    metadata,
                    "source_file",
                ),
                "file_type": safe_get(
                    metadata,
                    "file_type",
                ),
                "page_numbers": safe_get(
                    metadata,
                    "page_numbers",
                ),
                "page_number_start": (
                    safe_get(
                        metadata,
                        "page_number_start",
                    )
                ),
                "page_number_end": (
                    safe_get(
                        metadata,
                        "page_number_end",
                    )
                ),
                "slide_numbers": safe_get(
                    metadata,
                    "slide_numbers",
                ),
                "sheet_names": safe_get(
                    metadata,
                    "sheet_names",
                ),
                "sections": safe_get(
                    metadata,
                    "sections",
                ),
                "chunk_word_count": (
                    safe_get(
                        metadata,
                        "chunk_word_count",
                        0,
                    )
                ),
                "content_hash": safe_get(
                    metadata,
                    "content_hash",
                ),
                "content_preview": (
                    preview_text(
                        document
                    )
                ),
                "content": document,
            }
        )

    return rows


def deduplicate_rows(
    rows: Iterable[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    """
    Keep the best-distance occurrence of each chunk and
    preserve every query that matched the chunk.
    """
    best_by_chunk: dict[
        str,
        dict[str, Any],
    ] = {}

    matched_queries: dict[
        str,
        list[str],
    ] = {}

    for row in rows:
        chunk_id = str(
            row["chunk_id"]
        )

        query = str(
            row.get(
                "query",
                "",
            )
        )

        matched_queries.setdefault(
            chunk_id,
            [],
        )

        if (
            query
            and query
            not in matched_queries[
                chunk_id
            ]
        ):
            matched_queries[
                chunk_id
            ].append(query)

        current = best_by_chunk.get(
            chunk_id
        )

        if current is None:
            best_by_chunk[
                chunk_id
            ] = dict(row)
            continue

        current_distance = float(
            current.get(
                "distance",
                float("inf"),
            )
        )

        new_distance = float(
            row.get(
                "distance",
                float("inf"),
            )
        )

        if new_distance < current_distance:
            best_by_chunk[
                chunk_id
            ] = dict(row)

    deduplicated = []

    for chunk_id, row in (
        best_by_chunk.items()
    ):
        row["matched_queries"] = (
            matched_queries[
                chunk_id
            ]
        )

        deduplicated.append(row)

    deduplicated.sort(
        key=lambda row: float(
            row.get(
                "distance",
                float("inf"),
            )
        )
    )

    for rank, row in enumerate(
        deduplicated,
        start=1,
    ):
        row["combined_rank"] = rank

    return deduplicated


def retrieve_rfp_context(
    opportunity_id: str,
    queries: list[str] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Retrieve and deduplicate RFP context across a set of
    requirement-focused queries.
    """
    normalized_opportunity_id = str(
        opportunity_id
    ).strip()

    if not normalized_opportunity_id:
        raise ValueError(
            "Opportunity ID cannot be empty."
        )

    selected_queries = (
        queries
        if queries
        else DEFAULT_REQUIREMENT_QUERIES
    )

    cleaned_queries = [
        str(query).strip()
        for query in selected_queries
        if str(query).strip()
    ]

    if not cleaned_queries:
        raise ValueError(
            "At least one retrieval query is required."
        )

    collection = get_collection()

    all_rows: list[
        dict[str, Any]
    ] = []

    per_query_counts: dict[
        str,
        int,
    ] = {}

    for query in cleaned_queries:
        rows = retrieve_rfp_chunks(
            query=query,
            opportunity_id=(
                normalized_opportunity_id
            ),
            top_k=top_k,
            collection=collection,
        )

        per_query_counts[
            query
        ] = len(rows)

        all_rows.extend(rows)

    deduplicated_rows = (
        deduplicate_rows(
            all_rows
        )
    )

    return {
        "retrieval_type": (
            "rfp_requirement_context"
        ),
        "opportunity_id": (
            normalized_opportunity_id
        ),
        "database_target": "rfp_db",
        "collection_name": (
            RFP_COLLECTION
        ),
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "queries": cleaned_queries,
        "top_k_per_query": top_k,
        "per_query_counts": (
            per_query_counts
        ),
        "raw_result_count": len(
            all_rows
        ),
        "total_chunks": len(
            deduplicated_rows
        ),
        "chunks": (
            deduplicated_rows
        ),
    }


def print_results(
    rows: list[dict[str, Any]],
    max_preview: int = 10,
) -> None:
    print(
        "\nRetrieved unique chunks:",
        len(rows),
    )

    for row in rows[
        :max_preview
    ]:
        print(
            "\n" + "-" * 100
        )

        print(
            "Combined rank:",
            row.get(
                "combined_rank",
                row.get("rank"),
            ),
        )

        print(
            "Best query:",
            row.get("query"),
        )

        print(
            "Matched queries:",
            row.get(
                "matched_queries",
                [
                    row.get(
                        "query"
                    )
                ],
            ),
        )

        print(
            "Chunk ID:",
            row["chunk_id"],
        )

        print(
            "Distance:",
            row["distance"],
        )

        print(
            "Opportunity ID:",
            row["opportunity_id"],
        )

        print(
            "Database target:",
            row["database_target"],
        )

        print(
            "Collection:",
            row["collection_name"],
        )

        print(
            "Source file:",
            row["source_file"],
        )

        print(
            "File type:",
            row["file_type"],
        )

        if row["page_numbers"]:
            print(
                "Pages:",
                row["page_numbers"],
            )

        if row["sheet_names"]:
            print(
                "Sheets:",
                row["sheet_names"],
            )

        print("\nPreview:")
        print(
            row["content_preview"]
        )


def save_results(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(
            output_path
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink(
                missing_ok=True
            )

    print(
        "\nSaved RFP context to:",
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--opportunity-id",
        required=True,
        help=(
            "The opportunity_id to retrieve from."
        ),
    )

    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help=(
            "Optional query. Repeat --query to use "
            "multiple custom queries. Default requirement "
            "queries are used when omitted."
        ),
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

    parser.add_argument(
        "--max-preview",
        type=int,
        default=10,
        help=(
            "Maximum retrieved chunks printed "
            "to the terminal."
        ),
    )

    args = parser.parse_args()

    result = retrieve_rfp_context(
        opportunity_id=(
            args.opportunity_id
        ),
        queries=args.query,
        top_k=args.top_k,
    )

    print_results(
        result["chunks"],
        max_preview=args.max_preview,
    )

    save_results(
        result,
        Path(args.output),
    )


if __name__ == "__main__":
    main()
