from pathlib import Path
import json
import argparse
from typing import Any

import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


CHUNKS_PATH = Path("data/processed/chunks.jsonl")

RFP_DB_PATH = Path("data/chroma/rfp_db")
PROPOSAL_DB_PATH = Path("data/chroma/proposal_db")

RFP_COLLECTION = "rfp_documents"
PROPOSAL_COLLECTION = "proposal_knowledge"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def sanitize_metadata_value(value: Any):
    """
    Chroma metadata values must be simple scalar values:
    str, int, float, or bool.

    None, lists, dicts, and NaN values are not safe.
    """

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def build_metadata(chunk: dict) -> dict:
    metadata = {}

    for field in METADATA_FIELDS:
        metadata[field] = sanitize_metadata_value(chunk.get(field))

    return metadata


def batch_items(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def get_collection(db_path: Path, collection_name: str, embedding_function, reset: bool):
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))

    if reset:
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def insert_target_chunks(
    target_name: str,
    chunks: list[dict],
    embedding_function,
    reset: bool,
):
    config = TARGET_CONFIG[target_name]

    db_path = config["db_path"]
    expected_collection_name = config["collection_name"]

    target_chunks = [
        chunk for chunk in chunks
        if chunk.get("database_target") == target_name
    ]

    if not target_chunks:
        print(f"No chunks found for target: {target_name}")
        return 0

    invalid_collection_chunks = [
        chunk for chunk in target_chunks
        if chunk.get("collection_name") != expected_collection_name
    ]

    if invalid_collection_chunks:
        raise ValueError(
            f"Found chunks for {target_name} with wrong collection name. "
            f"Expected: {expected_collection_name}"
        )

    collection = get_collection(
        db_path=db_path,
        collection_name=expected_collection_name,
        embedding_function=embedding_function,
        reset=reset,
    )

    print("\n" + "=" * 80)
    print(f"Inserting target: {target_name}")
    print(f"DB path: {db_path}")
    print(f"Collection: {expected_collection_name}")
    print(f"Chunks: {len(target_chunks)}")
    print("=" * 80)

    for batch in tqdm(
        list(batch_items(target_chunks, BATCH_SIZE)),
        desc=f"Inserting {target_name}",
    ):
        ids = [chunk["chunk_id"] for chunk in batch]
        documents = [chunk["content"] for chunk in batch]
        metadatas = [build_metadata(chunk) for chunk in batch]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    final_count = collection.count()
    print(f"Final collection count for {expected_collection_name}: {final_count}")

    return final_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate Chroma collections before inserting.",
    )
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}. Run create_chunks.py first."
        )

    chunks = read_jsonl(CHUNKS_PATH)

    if not chunks:
        raise ValueError("No chunks found in chunks.jsonl")

    df = pd.DataFrame(chunks)

    print(f"Loaded chunks: {len(chunks)}")
    print("\nDatabase targets:")
    print(df["database_target"].value_counts(dropna=False))

    print("\nCollections:")
    print(df["collection_name"].value_counts(dropna=False))

    print("\nEmbedding model:")
    print(EMBEDDING_MODEL_NAME)

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    rfp_count = insert_target_chunks(
        target_name="rfp_db",
        chunks=chunks,
        embedding_function=embedding_function,
        reset=args.reset,
    )

    proposal_count = insert_target_chunks(
        target_name="proposal_db",
        chunks=chunks,
        embedding_function=embedding_function,
        reset=args.reset,
    )

    print("\nDone.")
    print(f"{RFP_COLLECTION}: {rfp_count}")
    print(f"{PROPOSAL_COLLECTION}: {proposal_count}")


if __name__ == "__main__":
    main()