"""Retrieve approved-company proposal knowledge from the proposal Chroma store."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any, Iterable

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


EXTRACTED_REQUIREMENTS_PATH = Path(
    "data/processed/extracted_requirements.json"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/processed/proposal_context"
)

PROPOSAL_DB_PATH = Path(
    "data/chroma/proposal_db"
)

PROPOSAL_COLLECTION = "proposal_knowledge"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


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


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load JSON input needed for proposal-knowledge retrieval."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Save retrieval context JSON through a temporary file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".part"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink(
                missing_ok=True
            )


def create_embedding_function(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformerEmbeddingFunction:
    """Create the embedding function used for proposal knowledge search."""

    return SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def get_collection(
    embedding_function=None,
):
    """Open the proposal knowledge Chroma collection."""

    if embedding_function is None:
        embedding_function = (
            create_embedding_function()
        )

    if not PROPOSAL_DB_PATH.exists():
        raise FileNotFoundError(
            "Proposal Chroma database not found: "
            f"{PROPOSAL_DB_PATH}"
        )

    client = chromadb.PersistentClient(
        path=str(PROPOSAL_DB_PATH)
    )

    try:
        return client.get_collection(
            name=PROPOSAL_COLLECTION,
            embedding_function=(
                embedding_function
            ),
        )
    except Exception as error:
        raise RuntimeError(
            "Proposal collection could not be opened: "
            f"{PROPOSAL_COLLECTION}"
        ) from error


def safe_get(
    metadata: dict[str, Any],
    key: str,
    default: Any = "",
) -> Any:
    """Return metadata values with a fallback for missing entries."""

    value = metadata.get(
        key,
        default,
    )

    if value is None:
        return default

    return value


def preview_text(
    text: str,
    max_chars: int = 450,
) -> str:
    """Create a compact preview for retrieved proposal chunks."""

    normalized = " ".join(
        (text or "").split()
    )

    if len(normalized) <= max_chars:
        return normalized

    return normalized[:max_chars] + "..."


def validate_proposal_type(
    proposal_type: str,
) -> str:
    """Normalize and validate the requested proposal type filter."""

    normalized = str(
        proposal_type
    ).strip().lower()

    if normalized not in {
        "business",
        "technical",
        "both",
    }:
        raise ValueError(
            "proposal_type must be business, "
            "technical, or both."
        )

    return normalized


def requirement_matches_proposal_type(
    requirement: dict[str, Any],
    proposal_type: str,
) -> bool:
    """Return whether an extracted requirement belongs in the proposal type."""

    proposal_type = validate_proposal_type(
        proposal_type
    )

    if proposal_type == "both":
        return True

    requirement_types = requirement.get(
        "proposal_type",
        [],
    )

    if not isinstance(
        requirement_types,
        list,
    ):
        requirement_types = [
            str(requirement_types)
        ]

    normalized_types = {
        str(value).strip().lower()
        for value in requirement_types
        if str(value).strip()
    }

    category = str(
        requirement.get(
            "category",
            "other",
        )
    ).strip().lower()

    if proposal_type == "business":
        return (
            "business" in normalized_types
            or category in BUSINESS_CATEGORIES
        )

    return (
        "technical" in normalized_types
        or category in TECHNICAL_CATEGORIES
    )


def select_requirements(
    requirements: list[dict[str, Any]],
    proposal_type: str,
    selected_requirement_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select requirements that should drive proposal knowledge retrieval."""

    proposal_type = validate_proposal_type(
        proposal_type
    )

    selected_id_set = {
        str(requirement_id).strip()
        for requirement_id
        in (
            selected_requirement_ids
            or []
        )
        if str(requirement_id).strip()
    }

    filtered = []

    for requirement in requirements:
        requirement_id = str(
            requirement.get(
                "requirement_id",
                "",
            )
        ).strip()

        if (
            selected_id_set
            and requirement_id
            not in selected_id_set
        ):
            continue

        if not requirement_matches_proposal_type(
            requirement,
            proposal_type,
        ):
            continue

        filtered.append(requirement)

    filtered.sort(
        key=lambda requirement: (
            0
            if requirement.get(
                "priority"
            )
            == "mandatory"
            else 1,
            str(
                requirement.get(
                    "requirement_id",
                    "",
                )
            ),
        )
    )

    return filtered


def build_query_from_requirement(
    requirement: dict[str, Any],
    proposal_type: str,
) -> str:
    """Build a semantic search query from one extracted requirement."""

    proposal_type = validate_proposal_type(
        proposal_type
    )

    requirement_text = str(
        requirement.get(
            "requirement_text",
            "",
        )
    ).strip()

    category = str(
        requirement.get(
            "category",
            "",
        )
    ).strip()

    priority = str(
        requirement.get(
            "priority",
            "",
        )
    ).strip()

    evidence = str(
        requirement.get(
            "evidence_quote",
            "",
        )
    ).strip()

    if proposal_type == "technical":
        intent = (
            "technical approach implementation methodology "
            "security architecture integration testing "
            "deployment solution delivery"
        )
    elif proposal_type == "business":
        intent = (
            "company experience business value delivery plan "
            "project management implementation timeline "
            "client outcomes"
        )
    else:
        intent = (
            "company experience implementation methodology "
            "technical approach business value project "
            "management solution delivery"
        )

    return f"""
{intent}

Requirement:
{requirement_text}

Category: {category}
Priority: {priority}
Supporting RFP evidence:
{evidence}
""".strip()


def add_general_queries(
    proposal_type: str,
) -> list[dict[str, str]]:
    """Build broad capability queries for the selected proposal type."""

    proposal_type = validate_proposal_type(
        proposal_type
    )

    if proposal_type == "technical":
        return [
            {
                "query": (
                    "technical approach implementation "
                    "methodology solution architecture "
                    "integration security testing deployment"
                ),
                "query_type": "general_technical",
                "requirement_id": (
                    "GENERAL-TECHNICAL"
                ),
            },
            {
                "query": (
                    "AI software implementation data privacy "
                    "security governance testing operations "
                    "and support"
                ),
                "query_type": "general_technical",
                "requirement_id": (
                    "GENERAL-TECHNICAL"
                ),
            },
        ]

    if proposal_type == "business":
        return [
            {
                "query": (
                    "company experience project management "
                    "delivery plan business value "
                    "implementation timeline client outcomes"
                ),
                "query_type": "general_business",
                "requirement_id": (
                    "GENERAL-BUSINESS"
                ),
            },
            {
                "query": (
                    "proposal executive summary relevant "
                    "experience delivery methodology "
                    "governance and customer success"
                ),
                "query_type": "general_business",
                "requirement_id": (
                    "GENERAL-BUSINESS"
                ),
            },
        ]

    return [
        {
            "query": (
                "company experience implementation methodology "
                "technical approach business value project "
                "management and delivery plan"
            ),
            "query_type": "general_both",
            "requirement_id": "GENERAL-BOTH",
        },
        {
            "query": (
                "AI solution architecture implementation "
                "security data governance delivery management "
                "and client outcomes"
            ),
            "query_type": "general_both",
            "requirement_id": "GENERAL-BOTH",
        },
    ]


def build_queries(
    selected_requirements: list[dict[str, Any]],
    proposal_type: str,
    max_requirement_queries: int = 12,
    include_general_queries: bool = True,
) -> list[dict[str, str]]:
    """Build the ordered query set for proposal knowledge retrieval."""

    if max_requirement_queries < 0:
        raise ValueError(
            "max_requirement_queries cannot be negative."
        )

    queries = []

    for requirement in selected_requirements[
        :max_requirement_queries
    ]:
        queries.append(
            {
                "query": (
                    build_query_from_requirement(
                        requirement,
                        proposal_type,
                    )
                ),
                "query_type": "requirement",
                "requirement_id": str(
                    requirement.get(
                        "requirement_id",
                        "",
                    )
                ),
            }
        )

    if include_general_queries:
        queries.extend(
            add_general_queries(
                proposal_type
            )
        )

    if not queries:
        raise ValueError(
            "No proposal retrieval queries were created."
        )

    return queries


def query_proposal_db(
    query: str,
    top_k: int = 5,
    collection=None,
) -> list[dict[str, Any]]:
    """Run one proposal-knowledge query and convert matches into rows."""

    normalized_query = str(
        query
    ).strip()

    if not normalized_query:
        raise ValueError(
            "Proposal retrieval query cannot be empty."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    if collection is None:
        collection = get_collection()

    results = collection.query(
        query_texts=[
            normalized_query
        ],
        n_results=top_k,
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

    if len(
        {
            len(ids),
            len(documents),
            len(metadatas),
            len(distances),
        }
    ) != 1:
        raise RuntimeError(
            "Chroma returned mismatched proposal "
            "retrieval arrays."
        )

    rows = []

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

        database_target = str(
            safe_get(
                metadata,
                "database_target",
                "",
            )
        ).strip()

        collection_name = str(
            safe_get(
                metadata,
                "collection_name",
                "",
            )
        ).strip()

        if database_target != "proposal_db":
            raise RuntimeError(
                "Proposal retrieval scope violation: "
                "a non-proposal chunk was returned."
            )

        if (
            collection_name
            != PROPOSAL_COLLECTION
        ):
            raise RuntimeError(
                "Proposal retrieval scope violation: "
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
                    database_target
                ),
                "collection_name": (
                    collection_name
                ),
                "source_file": safe_get(
                    metadata,
                    "source_file",
                ),
                "file_type": safe_get(
                    metadata,
                    "file_type",
                ),
                "folder_name": safe_get(
                    metadata,
                    "folder_name",
                ),
                "opportunity_id": safe_get(
                    metadata,
                    "opportunity_id",
                ),
                "project_id": safe_get(
                    metadata,
                    "project_id",
                ),
                "project_name": safe_get(
                    metadata,
                    "project_name",
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
                "content_preview": (
                    preview_text(
                        document
                    )
                ),
                "content": document,
            }
        )

    return rows


def deduplicate_results(
    rows: Iterable[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    """Merge duplicate chunks while preserving matched-query context."""

    best_by_chunk = {}
    requirement_ids_by_chunk = {}
    query_types_by_chunk = {}
    queries_by_chunk = {}

    for row in rows:
        chunk_id = str(
            row["chunk_id"]
        )

        requirement_id = str(
            row.get(
                "matched_requirement_id",
                "",
            )
        )

        query_type = str(
            row.get(
                "query_type",
                "",
            )
        )

        query = str(
            row.get(
                "query",
                "",
            )
        )

        requirement_ids_by_chunk.setdefault(
            chunk_id,
            [],
        )

        query_types_by_chunk.setdefault(
            chunk_id,
            [],
        )

        queries_by_chunk.setdefault(
            chunk_id,
            [],
        )

        if (
            requirement_id
            and requirement_id
            not in requirement_ids_by_chunk[
                chunk_id
            ]
        ):
            requirement_ids_by_chunk[
                chunk_id
            ].append(
                requirement_id
            )

        if (
            query_type
            and query_type
            not in query_types_by_chunk[
                chunk_id
            ]
        ):
            query_types_by_chunk[
                chunk_id
            ].append(
                query_type
            )

        if (
            query
            and query
            not in queries_by_chunk[
                chunk_id
            ]
        ):
            queries_by_chunk[
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

        if float(
            row.get(
                "distance",
                float("inf"),
            )
        ) < float(
            current.get(
                "distance",
                float("inf"),
            )
        ):
            best_by_chunk[
                chunk_id
            ] = dict(row)

    deduplicated = []

    for chunk_id, row in (
        best_by_chunk.items()
    ):
        row[
            "matched_requirement_ids"
        ] = requirement_ids_by_chunk[
            chunk_id
        ]

        row[
            "matched_query_types"
        ] = query_types_by_chunk[
            chunk_id
        ]

        row[
            "matched_queries"
        ] = queries_by_chunk[
            chunk_id
        ]

        deduplicated.append(row)

    deduplicated.sort(
        key=lambda row: float(
            row.get(
                "distance",
                float("inf"),
            )
        )
    )

    for combined_rank, row in enumerate(
        deduplicated,
        start=1,
    ):
        row[
            "combined_rank"
        ] = combined_rank

    return deduplicated


def retrieve_proposal_context(
    extracted_requirements: dict[str, Any],
    proposal_type: str = "both",
    selected_requirement_ids: list[str] | None = None,
    top_k: int = 5,
    max_requirement_queries: int = 12,
    include_general_queries: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Retrieve proposal knowledge chunks for selected RFP requirements."""

    proposal_type = validate_proposal_type(
        proposal_type
    )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    opportunity_id = str(
        extracted_requirements.get(
            "opportunity_id",
            "",
        )
        or ""
    ).strip()

    requirements = (
        extracted_requirements.get(
            "requirements",
            []
        )
    )

    if not opportunity_id:
        raise ValueError(
            "Extracted requirements have no "
            "opportunity_id."
        )

    if (
        not isinstance(
            requirements,
            list,
        )
        or not requirements
    ):
        raise ValueError(
            "No requirements were supplied."
        )

    selected_requirements = (
        select_requirements(
            requirements=requirements,
            proposal_type=proposal_type,
            selected_requirement_ids=(
                selected_requirement_ids
            ),
        )
    )

    if not selected_requirements:
        raise ValueError(
            "No requirements matched the selected "
            "proposal type and requirement selection."
        )

    queries = build_queries(
        selected_requirements=(
            selected_requirements
        ),
        proposal_type=proposal_type,
        max_requirement_queries=(
            max_requirement_queries
        ),
        include_general_queries=(
            include_general_queries
        ),
    )

    collection = get_collection()

    all_rows = []

    for query_item in queries:
        rows = query_proposal_db(
            query=query_item["query"],
            top_k=top_k,
            collection=collection,
        )

        for row in rows:
            row[
                "matched_requirement_id"
            ] = query_item[
                "requirement_id"
            ]

            row[
                "query_type"
            ] = query_item[
                "query_type"
            ]

        all_rows.extend(rows)

    deduplicated_rows = (
        deduplicate_results(
            all_rows
        )
    )

    payload = {
        "retrieval_type": (
            "proposal_knowledge_context"
        ),
        "opportunity_id": (
            opportunity_id
        ),
        "proposal_type": (
            proposal_type
        ),
        "database_target": (
            "proposal_db"
        ),
        "collection_name": (
            PROPOSAL_COLLECTION
        ),
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "total_requirements": len(
            requirements
        ),
        "selected_requirements_count": len(
            selected_requirements
        ),
        "selected_requirement_ids": [
            requirement.get(
                "requirement_id"
            )
            for requirement in (
                selected_requirements
            )
        ],
        "queries_count": len(
            queries
        ),
        "top_k_per_query": top_k,
        "max_requirement_queries": (
            max_requirement_queries
        ),
        "include_general_queries": (
            include_general_queries
        ),
        "selected_requirements": (
            selected_requirements
        ),
        "queries": queries,
        "total_retrieved_chunks_before_dedup": (
            len(all_rows)
        ),
        "total_retrieved_chunks_after_dedup": (
            len(deduplicated_rows)
        ),
        "chunks": deduplicated_rows,
    }

    if output_path is not None:
        save_json(
            payload,
            output_path,
        )

    return payload


def print_results(
    rows: list[dict[str, Any]],
    max_preview: int = 10,
) -> None:
    """Print a preview of retrieved proposal knowledge chunks."""

    print(
        "\nRetrieved unique proposal chunks:",
        len(rows),
    )

    for row in rows[
        :max_preview
    ]:
        print(
            "\n" + ("-" * 100)
        )

        print(
            "Combined rank:",
            row.get(
                "combined_rank",
                row.get("rank"),
            ),
        )

        print(
            "Matched requirements:",
            row.get(
                "matched_requirement_ids",
                [],
            ),
        )

        print(
            "Query types:",
            row.get(
                "matched_query_types",
                [],
            ),
        )

        print(
            "Distance:",
            row.get(
                "distance"
            ),
        )

        print(
            "Chunk ID:",
            row.get(
                "chunk_id"
            ),
        )

        print(
            "Database target:",
            row.get(
                "database_target"
            ),
        )

        print(
            "Collection:",
            row.get(
                "collection_name"
            ),
        )

        print(
            "Source file:",
            row.get(
                "source_file"
            ),
        )

        print(
            "Project name:",
            row.get(
                "project_name"
            ),
        )

        if row.get(
            "page_numbers"
        ):
            print(
                "Pages:",
                row[
                    "page_numbers"
                ],
            )

        print("\nPreview:")
        print(
            row.get(
                "content_preview",
                "",
            )
        )


def main() -> None:
    """CLI entry point for proposal-knowledge retrieval."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--requirements",
        default=str(
            EXTRACTED_REQUIREMENTS_PATH
        ),
    )

    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional output path. When omitted, an "
            "opportunity-specific path is used."
        ),
    )

    parser.add_argument(
        "--proposal-type",
        choices=[
            "business",
            "technical",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--requirement-id",
        action="append",
        default=None,
        help=(
            "Optional selected requirement ID. Repeat "
            "this flag to select multiple requirements."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-requirement-queries",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--no-general-queries",
        action="store_true",
    )

    parser.add_argument(
        "--max-preview",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    extracted_requirements = load_json(
        Path(
            args.requirements
        )
    )

    opportunity_id = str(
        extracted_requirements.get(
            "opportunity_id",
            "unknown",
        )
    )

    output_path = (
        Path(args.output)
        if args.output
        else (
            DEFAULT_OUTPUT_DIR
            / (
                f"{opportunity_id}_"
                f"{args.proposal_type}_"
                "proposal_context.json"
            )
        )
    )

    payload = retrieve_proposal_context(
        extracted_requirements=(
            extracted_requirements
        ),
        proposal_type=(
            args.proposal_type
        ),
        selected_requirement_ids=(
            args.requirement_id
        ),
        top_k=args.top_k,
        max_requirement_queries=(
            args.max_requirement_queries
        ),
        include_general_queries=(
            not args.no_general_queries
        ),
        output_path=output_path,
    )

    print(
        "Opportunity ID:",
        payload[
            "opportunity_id"
        ],
    )

    print(
        "Proposal type:",
        payload[
            "proposal_type"
        ],
    )

    print(
        "Total requirements:",
        payload[
            "total_requirements"
        ],
    )

    print(
        "Selected requirements:",
        payload[
            "selected_requirements_count"
        ],
    )

    print(
        "Queries:",
        payload[
            "queries_count"
        ],
    )

    print(
        "Top-k per query:",
        payload[
            "top_k_per_query"
        ],
    )

    print_results(
        payload["chunks"],
        max_preview=args.max_preview,
    )

    print(
        "\nSaved retrieved proposal context to:",
        output_path,
    )


if __name__ == "__main__":
    main()
