from pathlib import Path
import argparse
import csv
import json
import re
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


GOLDEN_PATH = Path("data/evaluation/golden_requirements.json")

RESULTS_CSV_PATH = Path("data/evaluation/retrieval_results.csv")
SUMMARY_JSON_PATH = Path("data/evaluation/retrieval_summary.json")

RFP_DB_PATH = Path("data/chroma/rfp_db")
PROPOSAL_DB_PATH = Path("data/chroma/proposal_db")

RFP_COLLECTION = "rfp_documents"
PROPOSAL_COLLECTION = "proposal_knowledge"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No evaluation rows were created.")

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def build_collections():
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    rfp_client = chromadb.PersistentClient(
        path=str(RFP_DB_PATH)
    )

    proposal_client = chromadb.PersistentClient(
        path=str(PROPOSAL_DB_PATH)
    )

    rfp_collection = rfp_client.get_collection(
        name=RFP_COLLECTION,
        embedding_function=embedding_function,
    )

    proposal_collection = proposal_client.get_collection(
        name=PROPOSAL_COLLECTION,
        embedding_function=embedding_function,
    )

    return {
        "rfp": rfp_collection,
        "proposal": proposal_collection,
    }


def determine_target(test_case: dict[str, Any]) -> str:
    explicit_target = test_case.get("target")

    if explicit_target in {"rfp", "proposal"}:
        return explicit_target

    category = test_case.get("category", "")

    if category == "proposal_knowledge":
        return "proposal"

    return "rfp"


def normalize_source(value: Any) -> str:
    return str(value or "").strip().lower()


def metadata_contains_page(
    metadata: dict[str, Any],
    expected_page: Any,
) -> bool:
    expected = str(expected_page or "").strip()

    if not expected:
        return False

    possible_values = [
        metadata.get("page_numbers"),
        metadata.get("page_number_start"),
        metadata.get("page_number_end"),
    ]

    for value in possible_values:
        if value is None:
            continue

        value_text = str(value).strip()

        if value_text == expected:
            return True

        page_tokens = re.findall(r"\d+", value_text)

        if expected in page_tokens:
            return True

    return False


def run_query(
    collection,
    question: str,
    top_k: int,
    opportunity_id: str | None = None,
) -> dict[str, Any]:
    n_results = min(top_k, collection.count())

    query_arguments = {
        "query_texts": [question],
        "n_results": n_results,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if opportunity_id:
        query_arguments["where"] = {
            "opportunity_id": opportunity_id
        }

    return collection.query(**query_arguments)


def first_relevant_rank(
    retrieved_ids: list[str],
    expected_ids: set[str],
) -> int | None:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected_ids:
            return rank

    return None


def evaluate_test_case(
    test_case: dict[str, Any],
    collections: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    target = determine_target(test_case)
    collection = collections[target]

    opportunity_id = None

    if target == "rfp":
        opportunity_id = test_case.get("opportunity_id")

    results = run_query(
        collection=collection,
        question=test_case["question"],
        top_k=top_k,
        opportunity_id=opportunity_id,
    )

    retrieved_ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    expected_ids = set(
        test_case.get("expected_chunk_ids", [])
    )

    matched_ids = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in expected_ids
    ]

    relevant_retrieved = len(matched_ids)
    retrieved_count = len(retrieved_ids)
    expected_count = len(expected_ids)

    precision_at_k = (
        relevant_retrieved / retrieved_count
        if retrieved_count
        else 0.0
    )

    recall_at_k = (
        relevant_retrieved / expected_count
        if expected_count
        else 0.0
    )

    hit_at_k = 1 if relevant_retrieved > 0 else 0

    first_rank = first_relevant_rank(
        retrieved_ids=retrieved_ids,
        expected_ids=expected_ids,
    )

    reciprocal_rank = (
        1.0 / first_rank
        if first_rank is not None
        else 0.0
    )

    top_1_chunk_accuracy = (
        1
        if retrieved_ids
        and retrieved_ids[0] in expected_ids
        else 0
    )

    expected_source = normalize_source(
        test_case.get("expected_source_file")
    )

    source_match = 0

    for metadata in metadatas:
        source_file = normalize_source(
            metadata.get("source_file")
        )

        if source_file == expected_source:
            source_match = 1
            break
################## Updated fixing the page issue 
    expected_page = str(
        test_case.get("expected_page") or ""
    ).strip()

    page_applicable = 1 if expected_page else 0

    page_match = None
    source_and_page_match = None

    if page_applicable:
        page_match = 0

        for metadata in metadatas:
            if metadata_contains_page(
                metadata=metadata,
                expected_page=expected_page,
            ):
                page_match = 1
                break

        source_and_page_match = 0

        for metadata in metadatas:
            source_file = normalize_source(
                metadata.get("source_file")
            )

            if (
                source_file == expected_source
                and metadata_contains_page(
                    metadata=metadata,
                    expected_page=expected_page,
                )
            ):
                source_and_page_match = 1
                break
###########################
    top_1_distance = (
        float(distances[0])
        if distances
        else None
    )

    retrieved_sources = [
        metadata.get("source_file", "")
        for metadata in metadatas
    ]

    retrieved_pages = [
        metadata.get("page_numbers", "")
        for metadata in metadatas
    ]

    return {
        "test_id": test_case.get("test_id"),
        "target": target,
        "question": test_case.get("question"),
        "category": test_case.get("category"),
        "top_k": top_k,
        "expected_chunk_ids": json.dumps(
            sorted(expected_ids),
            ensure_ascii=False,
        ),
        "retrieved_chunk_ids": json.dumps(
            retrieved_ids,
            ensure_ascii=False,
        ),
        "matched_chunk_ids": json.dumps(
            matched_ids,
            ensure_ascii=False,
        ),
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "hit_at_k": hit_at_k,
        "first_relevant_rank": first_rank or "",
        "reciprocal_rank": round(reciprocal_rank, 4),
        "top_1_chunk_accuracy": top_1_chunk_accuracy,
        "source_accuracy_at_k": source_match,
        "page_applicable": page_applicable,
        "page_accuracy_at_k": (
            page_match if page_match is not None else ""
        ),
        "source_and_page_accuracy_at_k": (
            source_and_page_match
            if source_and_page_match is not None
            else ""
        ),
        "top_1_distance": (
            round(top_1_distance, 6)
            if top_1_distance is not None
            else ""
        ),
        "expected_source_file": (
            test_case.get("expected_source_file", "")
        ),
        "expected_page": expected_page,
        "retrieved_sources": json.dumps(
            retrieved_sources,
            ensure_ascii=False,
        ),
        "retrieved_pages": json.dumps(
            retrieved_pages,
            ensure_ascii=False,
        ),
    }


def average(
    rows: list[dict[str, Any]],
    field: str,
) -> float | None:
    values = []

    for row in rows:
        value = row.get(field)

        if value in ("", None):
            continue

        values.append(float(value))

    if not values:
        return None

    return round(sum(values) / len(values), 4)


def build_summary(
    rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    rfp_rows = [
        row for row in rows
        if row["target"] == "rfp"
    ]

    proposal_rows = [
        row for row in rows
        if row["target"] == "proposal"
    ]

    def metric_summary(selected_rows):
        return {
            "test_count": len(selected_rows),
            "mean_precision_at_k": average(
                selected_rows,
                "precision_at_k",
            ),
            "mean_recall_at_k": average(
                selected_rows,
                "recall_at_k",
            ),
            "hit_rate_at_k": average(
                selected_rows,
                "hit_at_k",
            ),
            "mean_reciprocal_rank": average(
                selected_rows,
                "reciprocal_rank",
            ),
            "top_1_chunk_accuracy": average(
                selected_rows,
                "top_1_chunk_accuracy",
            ),
            "source_accuracy_at_k": average(
                selected_rows,
                "source_accuracy_at_k",
            ),
            "page_accuracy_at_k": average(
                selected_rows,
                "page_accuracy_at_k",
            ),
            "source_and_page_accuracy_at_k": average(
                selected_rows,
                "source_and_page_accuracy_at_k",
            ),
        }

    return {
        "evaluation_type": "golden_retrieval_evaluation",
        "top_k": top_k,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "overall": metric_summary(rows),
        "rfp_retrieval": metric_summary(rfp_rows),
        "proposal_retrieval": metric_summary(
            proposal_rows
        ),
        "metric_notes": {
            "precision_at_k": (
                "Exact gold chunk matches divided by retrieved chunks."
            ),
            "recall_at_k": (
                "Exact gold chunk matches divided by expected gold chunks."
            ),
            "hit_rate_at_k": (
                "Percentage of tests where at least one expected "
                "chunk appeared in top-k."
            ),
            "mean_reciprocal_rank": (
                "Rewards expected chunks appearing near rank one."
            ),
            "source_accuracy_at_k": (
                "Whether the expected source file appeared in top-k."
            ),
            "page_accuracy_at_k": (
                "Whether the expected page appeared in top-k metadata."
            ),
            "limitation": (
                "Precision uses only manually labeled expected chunk IDs. "
                "Other retrieved chunks may also be relevant but unlabeled."
            ),
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]

    print("\nRetrieval evaluation complete.")
    print(f"Top-k: {summary['top_k']}")
    print(f"Tests: {overall['test_count']}")

    print("\nOverall metrics:")
    print(
        "Mean Precision@K:",
        overall["mean_precision_at_k"],
    )
    print(
        "Mean Recall@K:",
        overall["mean_recall_at_k"],
    )
    print(
        "Hit Rate@K:",
        overall["hit_rate_at_k"],
    )
    print(
        "Mean Reciprocal Rank:",
        overall["mean_reciprocal_rank"],
    )
    print(
        "Top-1 Chunk Accuracy:",
        overall["top_1_chunk_accuracy"],
    )
    print(
        "Source Accuracy@K:",
        overall["source_accuracy_at_k"],
    )
    print(
        "Page Accuracy@K:",
        overall["page_accuracy_at_k"],
    )
    print(
        "Source + Page Accuracy@K:",
        overall["source_and_page_accuracy_at_k"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--golden",
        default=str(GOLDEN_PATH),
        help="Path to the golden evaluation JSON.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks per test.",
    )

    parser.add_argument(
        "--results-csv",
        default=str(RESULTS_CSV_PATH),
        help="Path for detailed CSV results.",
    )

    parser.add_argument(
        "--summary-json",
        default=str(SUMMARY_JSON_PATH),
        help="Path for summary JSON.",
    )

    args = parser.parse_args()

    golden_cases = load_json(Path(args.golden))

    if not isinstance(golden_cases, list):
        raise ValueError(
            "Golden evaluation file must contain a JSON list."
        )

    if not golden_cases:
        raise ValueError("Golden evaluation file is empty.")

    collections = build_collections()

    rows = []

    for test_case in golden_cases:
        print(
            f"Evaluating {test_case.get('test_id')}: "
            f"{test_case.get('question')}"
        )

        row = evaluate_test_case(
            test_case=test_case,
            collections=collections,
            top_k=args.top_k,
        )

        rows.append(row)

    summary = build_summary(
        rows=rows,
        top_k=args.top_k,
    )

    save_csv(
        rows=rows,
        path=Path(args.results_csv),
    )

    save_json(
        data=summary,
        path=Path(args.summary_json),
    )

    print_summary(summary)

    print(
        f"\nDetailed results saved to: {args.results_csv}"
    )

    print(
        f"Summary saved to: {args.summary_json}"
    )


if __name__ == "__main__":
    main()