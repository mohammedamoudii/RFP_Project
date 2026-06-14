from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import json
from typing import Any

import chromadb


RFP_DB_PATH = Path(
    "data/chroma/rfp_db"
)

RFP_COLLECTION = "rfp_documents"

REQUIREMENTS_DIR = Path(
    "data/processed/requirements"
)


def get_rfp_collection(
    db_path: Path = RFP_DB_PATH,
    collection_name: str = RFP_COLLECTION,
):
    if not db_path.exists():
        raise FileNotFoundError(
            f"RFP Chroma database not found: {db_path}"
        )

    client = chromadb.PersistentClient(
        path=str(db_path)
    )

    try:
        return client.get_collection(
            name=collection_name
        )
    except Exception as error:
        raise RuntimeError(
            "RFP collection could not be opened: "
            f"{collection_name}"
        ) from error


def get_requirements_path(
    opportunity_id: str,
    requirements_dir: Path = REQUIREMENTS_DIR,
) -> Path:
    return (
        requirements_dir
        / (
            f"{opportunity_id}_"
            "extracted_requirements.json"
        )
    )


def load_requirement_count(
    requirements_path: Path,
) -> int:
    if not requirements_path.exists():
        return 0

    try:
        with requirements_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        requirements = payload.get(
            "requirements",
            [],
        )

        return (
            len(requirements)
            if isinstance(requirements, list)
            else 0
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return 0


def list_rfp_opportunities(
    collection=None,
    requirements_dir: Path = REQUIREMENTS_DIR,
) -> list[dict[str, Any]]:
    """
    Return one structured record per RFP opportunity.

    The function reads metadata only. It does not load an
    embedding model and does not perform semantic retrieval.
    """
    if collection is None:
        collection = get_rfp_collection()

    data = collection.get(
        include=["metadatas"]
    )

    metadatas = (
        data.get(
            "metadatas",
            [],
        )
        or []
    )

    grouped: dict[
        str,
        dict[str, Any],
    ] = defaultdict(
        lambda: {
            "chunk_count": 0,
            "folder_names": set(),
            "source_files": set(),
            "file_types": set(),
            "collection_names": set(),
            "database_targets": set(),
        }
    )

    for metadata in metadatas:
        metadata = metadata or {}

        opportunity_id = str(
            metadata.get(
                "opportunity_id",
                "",
            )
            or ""
        ).strip()

        if not opportunity_id:
            continue

        record = grouped[
            opportunity_id
        ]

        record[
            "chunk_count"
        ] += 1

        for field, target_key in [
            (
                "folder_name",
                "folder_names",
            ),
            (
                "source_file",
                "source_files",
            ),
            (
                "file_type",
                "file_types",
            ),
            (
                "collection_name",
                "collection_names",
            ),
            (
                "database_target",
                "database_targets",
            ),
        ]:
            value = str(
                metadata.get(
                    field,
                    "",
                )
                or ""
            ).strip()

            if value:
                record[
                    target_key
                ].add(value)

    opportunities = []

    for opportunity_id, record in (
        grouped.items()
    ):
        requirements_path = (
            get_requirements_path(
                opportunity_id=(
                    opportunity_id
                ),
                requirements_dir=(
                    requirements_dir
                ),
            )
        )

        opportunities.append(
            {
                "opportunity_id": (
                    opportunity_id
                ),
                "chunk_count": (
                    record[
                        "chunk_count"
                    ]
                ),
                "folder_names": sorted(
                    record[
                        "folder_names"
                    ]
                ),
                "source_files": sorted(
                    record[
                        "source_files"
                    ]
                ),
                "file_types": sorted(
                    record[
                        "file_types"
                    ]
                ),
                "database_targets": sorted(
                    record[
                        "database_targets"
                    ]
                ),
                "collection_names": sorted(
                    record[
                        "collection_names"
                    ]
                ),
                "requirements_available": (
                    requirements_path.exists()
                ),
                "requirement_count": (
                    load_requirement_count(
                        requirements_path
                    )
                ),
                "requirements_path": str(
                    requirements_path
                ),
            }
        )

    opportunities.sort(
        key=lambda row: (
            not row[
                "requirements_available"
            ],
            -int(
                row[
                    "chunk_count"
                ]
            ),
            row[
                "opportunity_id"
            ],
        )
    )

    return opportunities


def print_opportunities(
    opportunities: list[
        dict[str, Any]
    ],
) -> None:
    total_chunks = sum(
        int(
            opportunity[
                "chunk_count"
            ]
        )
        for opportunity in opportunities
    )

    print(
        "Total RFP chunks:",
        total_chunks,
    )

    print(
        "Unique RFP opportunities:",
        len(opportunities),
    )

    print(
        "\nRFP opportunities:\n"
    )

    for opportunity in opportunities:
        print(
            f"{opportunity['opportunity_id']} "
            f"| chunks: "
            f"{opportunity['chunk_count']} "
            f"| requirements: "
            f"{opportunity['requirement_count']} "
            f"| requirement file: "
            f"{opportunity['requirements_available']} "
            f"| folders: "
            f"{', '.join(opportunity['folder_names']) or '-'} "
            f"| files: "
            f"{', '.join(opportunity['source_files']) or '-'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional JSON output path for the "
            "opportunity list."
        ),
    )

    args = parser.parse_args()

    opportunities = (
        list_rfp_opportunities()
    )

    print_opportunities(
        opportunities
    )

    if args.output:
        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "total_opportunities": (
                        len(opportunities)
                    ),
                    "opportunities": (
                        opportunities
                    ),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "\nSaved opportunity list to:",
            output_path,
        )


if __name__ == "__main__":
    main()
