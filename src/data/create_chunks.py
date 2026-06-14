from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import math
import re
from typing import Any

import pandas as pd
from tqdm import tqdm
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


INPUT_JSONL = Path(
    "data/processed/cleaned_documents.jsonl"
)
OUTPUT_JSONL = Path(
    "data/processed/chunks.jsonl"
)
REPORT_CSV = Path(
    "data/processed/chunking_report.csv"
)

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MIN_CHUNK_WORDS = 1


REPORT_COLUMNS = [
    "chunk_id",
    "opportunity_id",
    "project_id",
    "project_name",
    "database_target",
    "collection_name",
    "source_file",
    "file_type",
    "element_types",
    "page_numbers",
    "sheet_names",
    "chunk_char_count",
    "chunk_word_count",
    "is_short_chunk",
]


def build_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for index, line in enumerate(file):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)
            row["_input_order"] = index
            rows.append(row)

    return rows


def write_jsonl(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
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
            for row in rows:
                clean_row = {
                    key: value
                    for key, value in row.items()
                    if key != "_input_order"
                }

                file.write(
                    json.dumps(
                        clean_row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink(
                missing_ok=True
            )


def write_report_csv(
    chunks: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_rows = [
        {
            "chunk_id": chunk.get(
                "chunk_id"
            ),
            "opportunity_id": chunk.get(
                "opportunity_id"
            ),
            "project_id": chunk.get(
                "project_id"
            ),
            "project_name": chunk.get(
                "project_name"
            ),
            "database_target": chunk.get(
                "database_target"
            ),
            "collection_name": chunk.get(
                "collection_name"
            ),
            "source_file": chunk.get(
                "source_file"
            ),
            "file_type": chunk.get(
                "file_type"
            ),
            "element_types": chunk.get(
                "element_types"
            ),
            "page_numbers": chunk.get(
                "page_numbers"
            ),
            "sheet_names": chunk.get(
                "sheet_names"
            ),
            "chunk_char_count": chunk.get(
                "chunk_char_count"
            ),
            "chunk_word_count": chunk.get(
                "chunk_word_count"
            ),
            "is_short_chunk": chunk.get(
                "is_short_chunk"
            ),
        }
        for chunk in chunks
    ]

    dataframe = pd.DataFrame(
        report_rows,
        columns=REPORT_COLUMNS,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".part"
    )

    try:
        dataframe.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
        )

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink(
                missing_ok=True
            )


def is_missing(
    value: Any,
) -> bool:
    if value is None:
        return True

    if (
        isinstance(value, float)
        and math.isnan(value)
    ):
        return True

    return str(value).strip() == ""


def normalize_scalar(
    value: Any,
) -> Any:
    if is_missing(value):
        return None

    if (
        isinstance(value, float)
        and value.is_integer()
    ):
        return int(value)

    return value


def content_hash(
    text: str,
) -> str:
    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()[:16]


def count_words(
    text: str,
) -> int:
    if not text:
        return 0

    return len(
        re.findall(
            r"\b\w+\b",
            text,
        )
    )


def compact_unique(
    values: list[Any],
) -> str | None:
    result: list[str] = []

    for value in values:
        if is_missing(value):
            continue

        if (
            isinstance(value, float)
            and value.is_integer()
        ):
            value = int(value)

        normalized = str(value).strip()

        if (
            normalized
            and normalized not in result
        ):
            result.append(normalized)

    return (
        ", ".join(result)
        if result
        else None
    )


def numeric_values(
    values: list[Any],
) -> list[int]:
    numbers: list[int] = []

    for value in values:
        if is_missing(value):
            continue

        try:
            numbers.append(
                int(float(value))
            )
        except (TypeError, ValueError):
            continue

    return numbers


def format_element_text(
    row: dict[str, Any],
) -> str:
    content = str(
        row.get(
            "content",
            "",
        )
    ).strip()

    if not content:
        return ""

    element_type = row.get(
        "element_type"
    )
    page_number = row.get(
        "page_number"
    )
    slide_number = row.get(
        "slide_number"
    )
    sheet_name = row.get(
        "sheet_name"
    )
    row_start = row.get(
        "row_start"
    )
    row_end = row.get(
        "row_end"
    )
    section = row.get(
        "section"
    )

    prefix_parts: list[str] = []

    if not is_missing(section):
        prefix_parts.append(
            f"Section: {section}"
        )

    if (
        element_type == "pdf_page"
        and not is_missing(page_number)
    ):
        prefix_parts.append(
            f"Page: {page_number}"
        )

    if (
        element_type == "pptx_slide"
        and not is_missing(slide_number)
    ):
        prefix_parts.append(
            f"Slide: {slide_number}"
        )

    if element_type == "xlsx_sheet_rows":
        if not is_missing(sheet_name):
            prefix_parts.append(
                f"Sheet: {sheet_name}"
            )

        if (
            not is_missing(row_start)
            and not is_missing(row_end)
        ):
            prefix_parts.append(
                f"Rows: {row_start}-{row_end}"
            )

    if prefix_parts:
        return (
            "\n".join(prefix_parts)
            + "\n"
            + content
        )

    return content


def build_chunk_metadata(
    elements: list[dict[str, Any]],
    chunk_text: str,
    chunk_index: int,
) -> dict[str, Any]:
    first = elements[0]

    page_numbers = numeric_values(
        [
            element.get("page_number")
            for element in elements
        ]
    )

    slide_numbers = numeric_values(
        [
            element.get("slide_number")
            for element in elements
        ]
    )

    row_starts = numeric_values(
        [
            element.get("row_start")
            for element in elements
        ]
    )

    row_ends = numeric_values(
        [
            element.get("row_end")
            for element in elements
        ]
    )

    file_id = first.get(
        "file_id"
    )

    chunk_id_base = (
        f"{file_id}_"
        f"{chunk_index}_"
        f"{content_hash(chunk_text)}"
    )

    normalized_content = (
        chunk_text.strip()
    )

    return {
        "chunk_id": (
            f"chunk_{content_hash(chunk_id_base)}"
        ),
        "chunk_index": chunk_index,
        "doc_id": file_id,
        "file_id": file_id,
        "database_target": first.get(
            "database_target"
        ),
        "collection_name": first.get(
            "collection_name"
        ),
        "opportunity_id": normalize_scalar(
            first.get("opportunity_id")
        ),
        "project_id": normalize_scalar(
            first.get("project_id")
        ),
        "project_name": normalize_scalar(
            first.get("project_name")
        ),
        "folder_name": first.get(
            "folder_name"
        ),
        "source_file": first.get(
            "source_file"
        ),
        "file_type": first.get(
            "file_type"
        ),
        "relative_path": first.get(
            "relative_path"
        ),
        "absolute_path": first.get(
            "absolute_path"
        ),
        "element_ids": compact_unique(
            [
                element.get("element_id")
                for element in elements
            ]
        ),
        "element_types": compact_unique(
            [
                element.get("element_type")
                for element in elements
            ]
        ),
        "page_numbers": compact_unique(
            [
                element.get("page_number")
                for element in elements
            ]
        ),
        "page_number_start": (
            min(page_numbers)
            if page_numbers
            else None
        ),
        "page_number_end": (
            max(page_numbers)
            if page_numbers
            else None
        ),
        "slide_numbers": compact_unique(
            [
                element.get("slide_number")
                for element in elements
            ]
        ),
        "slide_number_start": (
            min(slide_numbers)
            if slide_numbers
            else None
        ),
        "slide_number_end": (
            max(slide_numbers)
            if slide_numbers
            else None
        ),
        "sheet_names": compact_unique(
            [
                element.get("sheet_name")
                for element in elements
            ]
        ),
        "row_start": (
            min(row_starts)
            if row_starts
            else None
        ),
        "row_end": (
            max(row_ends)
            if row_ends
            else None
        ),
        "sections": compact_unique(
            [
                element.get("section")
                for element in elements
            ]
        ),
        "content": normalized_content,
        "content_hash": content_hash(
            normalized_content
        ),
        "chunk_char_count": len(
            normalized_content
        ),
        "chunk_word_count": count_words(
            normalized_content
        ),
        "is_short_chunk": (
            count_words(
                normalized_content
            )
            < 20
        ),
    }


def add_split_texts_to_chunks(
    split_texts: list[str],
    elements: list[dict[str, Any]],
    file_id: str,
    chunk_counter: defaultdict[str, int],
    chunks: list[dict[str, Any]],
    minimum_chunk_words: int,
) -> None:
    for split_text in split_texts:
        split_text = split_text.strip()

        if not split_text:
            continue

        if (
            count_words(split_text)
            < minimum_chunk_words
        ):
            continue

        chunk_counter[file_id] += 1

        chunk = build_chunk_metadata(
            elements=elements,
            chunk_text=split_text,
            chunk_index=(
                chunk_counter[file_id]
            ),
        )

        chunks.append(chunk)


def flush_buffer(
    buffer_elements: list[dict[str, Any]],
    buffer_texts: list[str],
    chunk_counter: defaultdict[str, int],
    chunks: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
    minimum_chunk_words: int,
) -> None:
    if (
        not buffer_elements
        or not buffer_texts
    ):
        return

    full_text = "\n\n".join(
        buffer_texts
    ).strip()

    if not full_text:
        return

    if (
        count_words(full_text)
        < minimum_chunk_words
    ):
        return

    split_texts = splitter.split_text(
        full_text
    )

    file_id = str(
        buffer_elements[0].get(
            "file_id",
            "",
        )
    )

    add_split_texts_to_chunks(
        split_texts=split_texts,
        elements=buffer_elements,
        file_id=file_id,
        chunk_counter=chunk_counter,
        chunks=chunks,
        minimum_chunk_words=(
            minimum_chunk_words
        ),
    )


def create_chunks_from_elements(
    cleaned_elements: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    minimum_chunk_words: int = MIN_CHUNK_WORDS,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    if minimum_chunk_words < 1:
        raise ValueError(
            "minimum_chunk_words must be at least 1."
        )

    splitter = build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    rows = sorted(
        cleaned_elements,
        key=lambda row: (
            str(
                row.get(
                    "file_id",
                    "",
                )
            ),
            int(
                row.get(
                    "_input_order",
                    0,
                )
                or 0
            ),
        ),
    )

    grouped: defaultdict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        file_id = str(
            row.get(
                "file_id",
                "",
            )
        )

        if not file_id:
            continue

        grouped[file_id].append(row)

    chunks: list[dict[str, Any]] = []
    chunk_counter: defaultdict[
        str,
        int,
    ] = defaultdict(int)

    iterator = grouped.items()

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(grouped),
            desc="Creating chunks",
        )

    for file_id, file_elements in iterator:
        buffer_elements: list[
            dict[str, Any]
        ] = []
        buffer_texts: list[str] = []
        buffer_character_count = 0

        for element in file_elements:
            element_text = format_element_text(
                element
            )

            if not element_text:
                continue

            if (
                count_words(element_text)
                < minimum_chunk_words
            ):
                continue

            element_character_count = len(
                element_text
            )

            if (
                element_character_count
                > chunk_size
            ):
                flush_buffer(
                    buffer_elements=buffer_elements,
                    buffer_texts=buffer_texts,
                    chunk_counter=chunk_counter,
                    chunks=chunks,
                    splitter=splitter,
                    minimum_chunk_words=(
                        minimum_chunk_words
                    ),
                )

                buffer_elements = []
                buffer_texts = []
                buffer_character_count = 0

                split_texts = (
                    splitter.split_text(
                        element_text
                    )
                )

                add_split_texts_to_chunks(
                    split_texts=split_texts,
                    elements=[element],
                    file_id=file_id,
                    chunk_counter=chunk_counter,
                    chunks=chunks,
                    minimum_chunk_words=(
                        minimum_chunk_words
                    ),
                )

                continue

            if (
                buffer_texts
                and (
                    buffer_character_count
                    + element_character_count
                    > chunk_size
                )
            ):
                flush_buffer(
                    buffer_elements=buffer_elements,
                    buffer_texts=buffer_texts,
                    chunk_counter=chunk_counter,
                    chunks=chunks,
                    splitter=splitter,
                    minimum_chunk_words=(
                        minimum_chunk_words
                    ),
                )

                buffer_elements = []
                buffer_texts = []
                buffer_character_count = 0

            buffer_elements.append(
                element
            )

            buffer_texts.append(
                element_text
            )

            buffer_character_count += (
                element_character_count
            )

        flush_buffer(
            buffer_elements=buffer_elements,
            buffer_texts=buffer_texts,
            chunk_counter=chunk_counter,
            chunks=chunks,
            splitter=splitter,
            minimum_chunk_words=(
                minimum_chunk_words
            ),
        )

    return chunks


def belongs_to_scope(
    row: dict[str, Any],
    opportunity_id: str,
    database_target: str,
) -> bool:
    return (
        str(
            row.get(
                "opportunity_id",
                "",
            )
            or ""
        ).strip()
        == opportunity_id
        and str(
            row.get(
                "database_target",
                "",
            )
            or ""
        ).strip()
        == database_target
    )


def create_chunks_for_opportunity(
    cleaned_elements: list[dict[str, Any]],
    opportunity_id: str,
    database_target: str = "rfp_db",
    output_path: Path = OUTPUT_JSONL,
    report_path: Path = REPORT_CSV,
    replace_existing: bool = True,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    minimum_chunk_words: int = MIN_CHUNK_WORDS,
) -> dict[str, Any]:
    normalized_id = str(
        opportunity_id
    ).strip()

    if not normalized_id:
        raise ValueError(
            "Opportunity ID cannot be empty."
        )

    selected_elements = [
        row
        for row in cleaned_elements
        if belongs_to_scope(
            row,
            opportunity_id=normalized_id,
            database_target=database_target,
        )
    ]

    if not selected_elements:
        raise ValueError(
            "No cleaned elements matched "
            f"opportunity_id='{normalized_id}' and "
            f"database_target='{database_target}'."
        )

    new_chunks = create_chunks_from_elements(
        cleaned_elements=selected_elements,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        minimum_chunk_words=(
            minimum_chunk_words
        ),
        show_progress=False,
    )

    existing_chunks = read_jsonl(
        output_path
    )

    if replace_existing:
        existing_chunks = [
            row
            for row in existing_chunks
            if not belongs_to_scope(
                row,
                opportunity_id=normalized_id,
                database_target=database_target,
            )
        ]

    merged_chunks = (
        existing_chunks
        + new_chunks
    )

    write_jsonl(
        merged_chunks,
        output_path,
    )

    write_report_csv(
        merged_chunks,
        report_path,
    )

    return {
        "opportunity_id": normalized_id,
        "database_target": database_target,
        "input_element_count": len(
            selected_elements
        ),
        "chunk_count": len(
            new_chunks
        ),
        "chunks": new_chunks,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "minimum_chunk_words": (
            minimum_chunk_words
        ),
        "output_path": str(output_path),
        "report_path": str(report_path),
    }


def create_all_chunks(
    input_path: Path = INPUT_JSONL,
    output_path: Path = OUTPUT_JSONL,
    report_path: Path = REPORT_CSV,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    minimum_chunk_words: int = MIN_CHUNK_WORDS,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned file not found: {input_path}. "
            "Run clean_normalize.py first."
        )

    rows = read_jsonl(
        input_path
    )

    chunks = create_chunks_from_elements(
        cleaned_elements=rows,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        minimum_chunk_words=(
            minimum_chunk_words
        ),
        show_progress=True,
    )

    write_jsonl(
        chunks,
        output_path,
    )

    write_report_csv(
        chunks,
        report_path,
    )

    return {
        "input_element_count": len(rows),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "minimum_chunk_words": (
            minimum_chunk_words
        ),
        "output_path": str(output_path),
        "report_path": str(report_path),
    }


def print_summary(
    result: dict[str, Any],
) -> None:
    print(
        "Chunks saved to:",
        result["output_path"],
    )

    print(
        "Chunking report saved to:",
        result["report_path"],
    )

    print(
        "Input cleaned elements:",
        result["input_element_count"],
    )

    print(
        "Output chunks:",
        result["chunk_count"],
    )

    print(
        "Chunk size:",
        result["chunk_size"],
    )

    print(
        "Chunk overlap:",
        result["chunk_overlap"],
    )

    print(
        "Minimum chunk words:",
        result["minimum_chunk_words"],
    )

    chunks = result.get(
        "chunks",
        [],
    )

    if chunks:
        dataframe = pd.DataFrame(
            chunks
        )

        print("\nDatabase targets:")
        print(
            dataframe[
                "database_target"
            ].value_counts(
                dropna=False
            )
        )

        print("\nCollections:")
        print(
            dataframe[
                "collection_name"
            ].value_counts(
                dropna=False
            )
        )

        print("\nFile types:")
        print(
            dataframe[
                "file_type"
            ].value_counts(
                dropna=False
            )
        )

        print(
            "\nChunk word count summary:"
        )
        print(
            dataframe[
                "chunk_word_count"
            ].describe()
        )

        print("\nShort chunks:")
        print(
            dataframe[
                "is_short_chunk"
            ].value_counts(
                dropna=False
            )
        )

        print("\nZero-word chunks:")
        print(
            len(
                dataframe[
                    dataframe[
                        "chunk_word_count"
                    ]
                    == 0
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(INPUT_JSONL),
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_JSONL),
    )

    parser.add_argument(
        "--report",
        default=str(REPORT_CSV),
    )

    parser.add_argument(
        "--opportunity-id",
        default="",
    )

    parser.add_argument(
        "--database-target",
        default="rfp_db",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
    )

    parser.add_argument(
        "--minimum-chunk-words",
        type=int,
        default=MIN_CHUNK_WORDS,
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    report_path = Path(
        args.report
    )

    if args.opportunity_id:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Cleaned file not found: {input_path}. "
                "Run clean_normalize.py first."
            )

        cleaned_elements = read_jsonl(
            input_path
        )

        result = create_chunks_for_opportunity(
            cleaned_elements=cleaned_elements,
            opportunity_id=(
                args.opportunity_id
            ),
            database_target=(
                args.database_target
            ),
            output_path=output_path,
            report_path=report_path,
            replace_existing=True,
            chunk_size=args.chunk_size,
            chunk_overlap=(
                args.chunk_overlap
            ),
            minimum_chunk_words=(
                args.minimum_chunk_words
            ),
        )

    else:
        result = create_all_chunks(
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            chunk_size=args.chunk_size,
            chunk_overlap=(
                args.chunk_overlap
            ),
            minimum_chunk_words=(
                args.minimum_chunk_words
            ),
        )

    print_summary(result)


if __name__ == "__main__":
    main()
