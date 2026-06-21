"""Insert validated chunks into the separated RFP and proposal Chroma stores."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)

RFP_DB_PATH = Path(
    "data/chroma/rfp_db"
)

PROPOSAL_DB_PATH = Path(
    "data/chroma/proposal_db"
)

RFP_COLLECTION = "rfp_documents"
PROPOSAL_COLLECTION = "proposal_knowledge"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BATCH_SIZE = 128


TARGET_CONFIG = {
    "rfp_db": {
        "db_path": RFP_DB_PATH,
        "collection_name": RFP_COLLECTION,
    },
    "proposal_db": {
        "db_path": PROPOSAL_DB_PATH,
        "collection_name": PROPOSAL_COLLECTION,
    },
}


METADATA_FIELDS = [
    "chunk_id",
    "chunk_index",
    "doc_id",
    "file_id",
    "database_target",
    "collection_name",
    "opportunity_id",
    "project_id",
    "project_name",
    "folder_name",
    "source_file",
    "file_type",
    "relative_path",
    "element_types",
    "page_numbers",
    "page_number_start",
    "page_number_end",
    "slide_numbers",
    "slide_number_start",
    "slide_number_end",
    "sheet_names",
    "row_start",
    "row_end",
    "sections",
    "content_hash",
    "chunk_char_count",
    "chunk_word_count",
    "is_short_chunk",
]


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Read chunk records from a JSONL file."""

    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def sanitize_metadata_value(
    value: Any,
) -> str | int | float | bool:
    """
    Chroma metadata values must be scalar values.
    """

    if value is None:
        return ""

    if (
        isinstance(value, float)
        and pd.isna(value)
    ):
        return ""

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


def build_metadata(
    chunk: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """Build the Chroma-safe metadata payload for one chunk."""

    return {
        field: sanitize_metadata_value(
            chunk.get(field)
        )
        for field in METADATA_FIELDS
    }


def batch_items(
    items: list[dict[str, Any]],
    batch_size: int,
) -> Iterable[list[dict[str, Any]]]:
    """Yield fixed-size insertion batches."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start:start + batch_size
        ]


def create_embedding_function(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformerEmbeddingFunction:
    """Create the sentence-transformer embedding function used by Chroma."""

    return SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def get_collection(
    target_name: str,
    embedding_function,
    reset: bool = False,
):
    """Open or create the configured Chroma collection for a database target."""

    if target_name not in TARGET_CONFIG:
        raise ValueError(
            f"Unsupported database target: {target_name}"
        )

    config = TARGET_CONFIG[
        target_name
    ]

    db_path: Path = config[
        "db_path"
    ]

    collection_name: str = config[
        "collection_name"
    ]

    db_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(db_path)
    )

    if reset:
        try:
            client.delete_collection(
                name=collection_name
            )

            print(
                "Deleted existing collection:",
                collection_name,
            )

        except Exception:
            pass

    collection = (
        client.get_or_create_collection(
            name=collection_name,
            embedding_function=(
                embedding_function
            ),
            metadata={
                "hnsw:space": "cosine"
            },
        )
    )

    return collection


def validate_chunks(
    chunks: list[dict[str, Any]],
    target_name: str,
) -> None:
    """Validate chunk scope, IDs, collection name, and content before insertion."""

    if target_name not in TARGET_CONFIG:
        raise ValueError(
            f"Unsupported database target: {target_name}"
        )

    expected_collection = (
        TARGET_CONFIG[
            target_name
        ][
            "collection_name"
        ]
    )

    if not chunks:
        raise ValueError(
            "No chunks were supplied for insertion."
        )

    invalid_target = [
        chunk
        for chunk in chunks
        if chunk.get(
            "database_target"
        )
        != target_name
    ]

    if invalid_target:
        raise ValueError(
            "Some chunks have the wrong "
            f"database_target. Expected: {target_name}"
        )

    invalid_collection = [
        chunk
        for chunk in chunks
        if chunk.get(
            "collection_name"
        )
        != expected_collection
    ]

    if invalid_collection:
        raise ValueError(
            "Some chunks have the wrong collection "
            f"name. Expected: {expected_collection}"
        )

    missing_ids = [
        chunk
        for chunk in chunks
        if not str(
            chunk.get(
                "chunk_id",
                "",
            )
        ).strip()
    ]

    if missing_ids:
        raise ValueError(
            "Some chunks are missing chunk_id."
        )

    empty_documents = [
        chunk
        for chunk in chunks
        if not str(
            chunk.get(
                "content",
                "",
            )
        ).strip()
    ]

    if empty_documents:
        raise ValueError(
            "Some chunks contain empty content."
        )

    chunk_ids = [
        str(
            chunk["chunk_id"]
        )
        for chunk in chunks
    ]

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise ValueError(
            "Duplicate chunk IDs were found "
            "in the insertion batch."
        )


def upsert_chunks(
    collection,
    chunks: list[dict[str, Any]],
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> None:
    """Upsert chunk documents and metadata into Chroma in batches."""

    batches = list(
        batch_items(
            chunks,
            batch_size,
        )
    )

    iterator = batches

    if show_progress:
        iterator = tqdm(
            batches,
            desc="Embedding and inserting",
        )

    for batch in iterator:
        ids = [
            str(
                chunk["chunk_id"]
            )
            for chunk in batch
        ]

        documents = [
            str(
                chunk["content"]
            )
            for chunk in batch
        ]

        metadatas = [
            build_metadata(
                chunk
            )
            for chunk in batch
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )


def insert_opportunity_chunks(
    chunks: list[dict[str, Any]],
    opportunity_id: str,
    database_target: str = "rfp_db",
    embedding_function=None,
    replace_existing: bool = True,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Insert only one opportunity into its configured
    Chroma collection.

    When replace_existing=True, only records matching
    the active opportunity are deleted before insertion.
    Other opportunities and the proposal database are
    preserved.
    """

    normalized_id = str(
        opportunity_id
    ).strip()

    if not normalized_id:
        raise ValueError(
            "Opportunity ID cannot be empty."
        )

    if database_target != "rfp_db":
        raise ValueError(
            "Opportunity-scoped insertion is intended "
            "for database_target='rfp_db'."
        )

    selected_chunks = [
        chunk
        for chunk in chunks
        if (
            str(
                chunk.get(
                    "opportunity_id",
                    "",
                )
                or ""
            ).strip()
            == normalized_id
            and chunk.get(
                "database_target"
            )
            == database_target
        )
    ]

    validate_chunks(
        selected_chunks,
        target_name=database_target,
    )

    if embedding_function is None:
        embedding_function = (
            create_embedding_function()
        )

    collection = get_collection(
        target_name=database_target,
        embedding_function=(
            embedding_function
        ),
        reset=False,
    )

    existing_result = collection.get(
        where={
            "opportunity_id": normalized_id
        },
        include=[],
    )

    existing_ids = list(
        existing_result.get(
            "ids",
            [],
        )
    )

    if (
        replace_existing
        and existing_ids
    ):
        collection.delete(
            ids=existing_ids
        )

    upsert_chunks(
        collection=collection,
        chunks=selected_chunks,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    verification = collection.get(
        where={
            "opportunity_id": normalized_id
        },
        include=[],
    )

    stored_ids = list(
        verification.get(
            "ids",
            [],
        )
    )

    expected_ids = {
        str(
            chunk["chunk_id"]
        )
        for chunk in selected_chunks
    }

    actual_ids = set(
        stored_ids
    )

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    if missing_ids:
        raise RuntimeError(
            "Chroma verification failed. "
            f"Missing {len(missing_ids)} inserted chunk(s)."
        )

    return {
        "opportunity_id": normalized_id,
        "database_target": database_target,
        "collection_name": (
            TARGET_CONFIG[
                database_target
            ][
                "collection_name"
            ]
        ),
        "db_path": str(
            TARGET_CONFIG[
                database_target
            ][
                "db_path"
            ]
        ),
        "input_chunk_count": len(
            selected_chunks
        ),
        "previous_scope_count": len(
            existing_ids
        ),
        "stored_scope_count": len(
            stored_ids
        ),
        "collection_count": (
            collection.count()
        ),
        "replaced_existing": (
            replace_existing
        ),
        "chunk_ids": sorted(
            expected_ids
        ),
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
    }


def insert_target_chunks(
    target_name: str,
    chunks: list[dict[str, Any]],
    embedding_function,
    reset: bool,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Full-batch insertion retained for the CLI.
    """

    target_chunks = [
        chunk
        for chunk in chunks
        if chunk.get(
            "database_target"
        )
        == target_name
    ]

    if not target_chunks:
        print(
            f"No chunks found for target: {target_name}"
        )
        return 0

    validate_chunks(
        target_chunks,
        target_name=target_name,
    )

    collection = get_collection(
        target_name=target_name,
        embedding_function=(
            embedding_function
        ),
        reset=reset,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"Inserting target: {target_name}"
    )

    print(
        "DB path:",
        TARGET_CONFIG[
            target_name
        ][
            "db_path"
        ],
    )

    print(
        "Collection:",
        TARGET_CONFIG[
            target_name
        ][
            "collection_name"
        ],
    )

    print(
        "Chunks:",
        len(target_chunks),
    )

    print(
        "=" * 80
    )

    upsert_chunks(
        collection=collection,
        chunks=target_chunks,
        batch_size=batch_size,
        show_progress=True,
    )

    final_count = collection.count()

    print(
        "Final collection count:",
        final_count,
    )

    return final_count


def main() -> None:
    """CLI entry point for scoped or full Chroma insertion."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks",
        default=str(CHUNKS_PATH),
    )

    parser.add_argument(
        "--opportunity-id",
        default="",
    )

    parser.add_argument(
        "--database-target",
        default="rfp_db",
        choices=[
            "rfp_db",
            "proposal_db",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete and recreate complete collections. "
            "Use only for intentional full rebuilds."
        ),
    )

    args = parser.parse_args()

    chunks_path = Path(
        args.chunks
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}. "
            "Run create_chunks.py first."
        )

    chunks = read_jsonl(
        chunks_path
    )

    if not chunks:
        raise ValueError(
            "No chunks found in chunks.jsonl"
        )

    print(
        "Loaded chunks:",
        len(chunks),
    )

    dataframe = pd.DataFrame(
        chunks
    )

    print(
        "\nDatabase targets:"
    )

    print(
        dataframe[
            "database_target"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nCollections:"
    )

    print(
        dataframe[
            "collection_name"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nEmbedding model:"
    )

    print(
        EMBEDDING_MODEL_NAME
    )

    embedding_function = (
        create_embedding_function()
    )

    if args.opportunity_id:
        if args.reset:
            raise ValueError(
                "--reset cannot be combined with "
                "--opportunity-id because reset deletes "
                "the complete collection."
            )

        result = insert_opportunity_chunks(
            chunks=chunks,
            opportunity_id=(
                args.opportunity_id
            ),
            database_target=(
                args.database_target
            ),
            embedding_function=(
                embedding_function
            ),
            replace_existing=True,
            batch_size=args.batch_size,
            show_progress=True,
        )

        print(
            "\nOpportunity insertion complete."
        )

        for key, value in result.items():
            if key == "chunk_ids":
                continue

            print(
                f"{key}: {value}"
            )

        return

    rfp_count = insert_target_chunks(
        target_name="rfp_db",
        chunks=chunks,
        embedding_function=(
            embedding_function
        ),
        reset=args.reset,
        batch_size=args.batch_size,
    )

    proposal_count = insert_target_chunks(
        target_name="proposal_db",
        chunks=chunks,
        embedding_function=(
            embedding_function
        ),
        reset=args.reset,
        batch_size=args.batch_size,
    )

    print("\nDone.")

    print(
        f"{RFP_COLLECTION}: {rfp_count}"
    )

    print(
        f"{PROPOSAL_COLLECTION}: {proposal_count}"
    )


if __name__ == "__main__":
    main()
