"""Clean parsed document elements before chunking and embedding.

The parser output can contain page numbers, control characters, empty records,
and inconsistent whitespace. This module normalizes that text while preserving
the metadata needed to keep RFP content and proposal knowledge separated in the
downstream ChromaDB collections.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import re
from typing import Any

import pandas as pd


# -----------------------------------------------------------------------------
# Project paths and report schema
# -----------------------------------------------------------------------------
INPUT_JSONL = Path("data/processed/parsed_elements.jsonl")
OUTPUT_JSONL = Path("data/processed/cleaned_documents.jsonl")
REPORT_CSV = Path("data/processed/cleaning_report.csv")

REPORT_COLUMNS = [
    "element_id",
    "opportunity_id",
    "project_id",
    "database_target",
    "collection_name",
    "source_file",
    "file_type",
    "element_type",
    "original_length",
    "cleaned_length",
    "word_count",
    "kept",
]


# -----------------------------------------------------------------------------
# File IO helpers
# -----------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk, returning an empty list when absent."""
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write JSONL through a temporary file to avoid partial outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    """Read a CSV report into records, treating missing or empty files as none."""
    if not path.exists() or path.stat().st_size == 0:
        return []

    try:
        dataframe = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError:
        return []

    return dataframe.to_dict(orient="records")


def write_report_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the cleaning report CSV through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    temporary_path = path.with_suffix(path.suffix + ".part")

    try:
        dataframe.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def content_hash(text: str) -> str:
    """Create a stable short hash for cleaned content."""

    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def normalize_bullets(text: str) -> str:
    """Convert common PDF bullet glyphs and dashes into simple hyphens."""

    for bullet in ["•", "●", "○", "▪", "▫", "–", "—"]:
        text = text.replace(bullet, "-")
    return text


def remove_control_characters(text: str) -> str:
    """Remove non-printing control characters that can break downstream text."""

    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)


def fix_broken_hyphenation(text: str) -> str:
    """Join words split across line breaks by PDF hyphenation."""

    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Normalize line endings, repeated spaces, and excessive blank lines."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_noise_lines(text: str) -> str:
    """Drop standalone page-number lines while preserving document text."""

    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if re.fullmatch(r"\d{1,4}", stripped):
            continue

        if re.fullmatch(
            r"page\s+\d{1,4}(\s+of\s+\d{1,4})?",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


def clean_text(text: Any) -> str:
    """Apply all text cleanup passes to a parsed document element."""

    if text is None:
        return ""

    cleaned = str(text)
    cleaned = remove_control_characters(cleaned)
    cleaned = fix_broken_hyphenation(cleaned)
    cleaned = normalize_bullets(cleaned)
    cleaned = remove_page_noise_lines(cleaned)
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def count_words(text: str) -> int:
    """Count word-like tokens after cleaning."""

    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def clean_parsed_elements(
    parsed_elements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Clean parsed elements and build rows for the cleaning report."""

    cleaned_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for row in parsed_elements:
        original_content = row.get("content", "")
        cleaned_content = clean_text(original_content)

        original_length = (
            len(str(original_content))
            if original_content is not None
            else 0
        )
        cleaned_length = len(cleaned_content)
        word_count = count_words(cleaned_content)
        keep = cleaned_length > 0 and word_count > 0

        report_rows.append(
            {
                "element_id": row.get("element_id"),
                "opportunity_id": row.get("opportunity_id"),
                "project_id": row.get("project_id"),
                "database_target": row.get("database_target"),
                "collection_name": row.get("collection_name"),
                "source_file": row.get("source_file"),
                "file_type": row.get("file_type"),
                "element_type": row.get("element_type"),
                "original_length": original_length,
                "cleaned_length": cleaned_length,
                "word_count": word_count,
                "kept": keep,
            }
        )

        if not keep:
            continue

        cleaned_row = dict(row)
        cleaned_row["content"] = cleaned_content
        cleaned_row["content_hash"] = content_hash(cleaned_content)
        cleaned_row["char_count"] = cleaned_length
        cleaned_row["word_count"] = word_count
        cleaned_rows.append(cleaned_row)

    return cleaned_rows, report_rows


def belongs_to_scope(
    row: dict[str, Any],
    opportunity_id: str,
    database_target: str,
) -> bool:
    """Return whether a row belongs to one opportunity/database scope."""

    return (
        str(row.get("opportunity_id", "") or "").strip()
        == opportunity_id
        and str(row.get("database_target", "") or "").strip()
        == database_target
    )


def clean_opportunity_elements(
    parsed_elements: list[dict[str, Any]],
    opportunity_id: str,
    database_target: str = "rfp_db",
    output_path: Path = OUTPUT_JSONL,
    report_path: Path = REPORT_CSV,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """Clean and persist parsed elements for one uploaded RFP opportunity."""

    normalized_id = str(opportunity_id).strip()

    if not normalized_id:
        raise ValueError("Opportunity ID cannot be empty.")

    selected_elements = [
        row
        for row in parsed_elements
        if belongs_to_scope(
            row,
            opportunity_id=normalized_id,
            database_target=database_target,
        )
    ]

    if not selected_elements:
        raise ValueError(
            "No parsed elements matched "
            f"opportunity_id='{normalized_id}' and "
            f"database_target='{database_target}'."
        )

    cleaned_rows, report_rows = clean_parsed_elements(selected_elements)

    existing_cleaned = read_jsonl(output_path)
    existing_report = read_csv_records(report_path)

    if replace_existing:
        existing_cleaned = [
            row
            for row in existing_cleaned
            if not belongs_to_scope(
                row,
                opportunity_id=normalized_id,
                database_target=database_target,
            )
        ]
        existing_report = [
            row
            for row in existing_report
            if not belongs_to_scope(
                row,
                opportunity_id=normalized_id,
                database_target=database_target,
            )
        ]

    merged_cleaned = existing_cleaned + cleaned_rows
    merged_report = existing_report + report_rows

    write_jsonl(merged_cleaned, output_path)
    write_report_csv(merged_report, report_path)

    return {
        "opportunity_id": normalized_id,
        "database_target": database_target,
        "input_element_count": len(selected_elements),
        "cleaned_element_count": len(cleaned_rows),
        "removed_element_count": len(selected_elements) - len(cleaned_rows),
        "elements": cleaned_rows,
        "report_rows": report_rows,
        "output_path": str(output_path),
        "report_path": str(report_path),
    }


def clean_all_documents(
    input_path: Path = INPUT_JSONL,
    output_path: Path = OUTPUT_JSONL,
    report_path: Path = REPORT_CSV,
) -> dict[str, Any]:
    """Clean every parsed element in the shared processed JSONL file."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Parsed file not found: {input_path}. "
            "Run parse_documents.py first."
        )

    rows = read_jsonl(input_path)
    cleaned_rows, report_rows = clean_parsed_elements(rows)

    write_jsonl(cleaned_rows, output_path)
    write_report_csv(report_rows, report_path)

    return {
        "input_element_count": len(rows),
        "cleaned_element_count": len(cleaned_rows),
        "removed_element_count": len(rows) - len(cleaned_rows),
        "elements": cleaned_rows,
        "report_rows": report_rows,
        "output_path": str(output_path),
        "report_path": str(report_path),
    }


def print_summary(result: dict[str, Any]) -> None:
    """Print the cleaning counts and output paths for CLI runs."""

    print("Cleaned documents saved to:", result["output_path"])
    print("Cleaning report saved to:", result["report_path"])
    print("Input elements:", result["input_element_count"])
    print("Output cleaned elements:", result["cleaned_element_count"])
    print("Removed empty/no-word elements:", result["removed_element_count"])


def main() -> None:
    """CLI entry point for full or opportunity-scoped cleaning."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_JSONL))
    parser.add_argument("--output", default=str(OUTPUT_JSONL))
    parser.add_argument("--report", default=str(REPORT_CSV))
    parser.add_argument("--opportunity-id", default="")
    parser.add_argument("--database-target", default="rfp_db")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if args.opportunity_id:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Parsed file not found: {input_path}. "
                "Run parse_documents.py first."
            )

        parsed_elements = read_jsonl(input_path)
        result = clean_opportunity_elements(
            parsed_elements=parsed_elements,
            opportunity_id=args.opportunity_id,
            database_target=args.database_target,
            output_path=output_path,
            report_path=report_path,
            replace_existing=True,
        )
    else:
        result = clean_all_documents(
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
        )

    print_summary(result)


if __name__ == "__main__":
    main()
