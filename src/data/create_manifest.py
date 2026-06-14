from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any
import argparse
import hashlib

import pandas as pd


RAW_ROOT = Path("data/raw")
OUTPUT_PATH = Path("data/processed/file_manifest.csv")
SKIPPED_PATH = Path("data/processed/skipped_unknown_files.csv")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".md",
    ".txt",
    ".xlsx",
}

MANIFEST_COLUMNS = [
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
    "absolute_path",
    "file_size_bytes",
    "modified_time",
    "status",
    "error_message",
]

SKIPPED_COLUMNS = [
    "opportunity_id",
    "relative_path",
    "source_file",
    "file_type",
    "reason",
]


def make_id(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:12]


def normalize_id(text: str) -> str:
    normalized = (
        str(text)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def detect_database_target(
    relative_parts: tuple[str, ...],
) -> tuple[str, str]:
    if not relative_parts:
        return "unknown", "unknown"

    top_folder = relative_parts[0].lower()

    if top_folder == "rfp_uploads":
        return "rfp_db", "rfp_documents"

    if top_folder == "proposal_knowledge":
        return "proposal_db", "proposal_knowledge"

    return "unknown", "unknown"


def get_project_or_opportunity_name(
    relative_parts: tuple[str, ...],
) -> str:
    """
    Expected:
    data/raw/rfp_uploads/<opportunity_folder>/...
    data/raw/proposal_knowledge/<project_folder>/...
    """
    if len(relative_parts) >= 2:
        return relative_parts[1]

    return "unknown"


def _empty_manifest_df() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def _empty_skipped_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SKIPPED_COLUMNS)


def _read_csv_or_empty(
    path: Path,
    columns: list[str],
) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in frame.columns:
            frame[column] = ""

    return frame[columns]


def _build_manifest_row(
    file_path: Path,
    raw_root: Path,
    database_target: str,
    collection_name: str,
    folder_name: str,
    opportunity_id: str | None,
    project_id: str | None,
    project_name: str | None,
) -> dict[str, Any]:
    relative_path = file_path.relative_to(raw_root)
    relative_path_str = relative_path.as_posix()
    stat = file_path.stat()

    return {
        "file_id": make_id(relative_path_str),
        "database_target": database_target,
        "collection_name": collection_name,
        "opportunity_id": opportunity_id,
        "project_id": project_id,
        "project_name": project_name,
        "folder_name": folder_name,
        "source_file": file_path.name,
        "file_type": file_path.suffix.lower().replace(".", ""),
        "relative_path": relative_path_str,
        "absolute_path": str(file_path.resolve()),
        "file_size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(),
        "status": "pending",
        "error_message": "",
    }


def _save_manifest_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
    opportunity_id: str | None = None,
    merge_existing: bool = False,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    if merge_existing:
        existing_df = _read_csv_or_empty(
            output_path,
            MANIFEST_COLUMNS,
        )

        if opportunity_id:
            keep_mask = ~(
                (existing_df["database_target"] == "rfp_db")
                & (
                    existing_df["opportunity_id"]
                    == opportunity_id
                )
            )
            existing_df = existing_df.loc[keep_mask]

        combined_df = pd.concat(
            [existing_df, new_df],
            ignore_index=True,
        )
    else:
        combined_df = new_df

    combined_df = combined_df[MANIFEST_COLUMNS]
    combined_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return combined_df


def _save_skipped_rows(
    rows: list[dict[str, Any]],
    skipped_path: Path,
    opportunity_id: str | None = None,
    merge_existing: bool = False,
) -> pd.DataFrame:
    skipped_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows, columns=SKIPPED_COLUMNS)

    if merge_existing:
        existing_df = _read_csv_or_empty(
            skipped_path,
            SKIPPED_COLUMNS,
        )

        if opportunity_id:
            existing_df = existing_df.loc[
                existing_df["opportunity_id"]
                != opportunity_id
            ]

        combined_df = pd.concat(
            [existing_df, new_df],
            ignore_index=True,
        )
    else:
        combined_df = new_df

    combined_df = combined_df[SKIPPED_COLUMNS]
    combined_df.to_csv(
        skipped_path,
        index=False,
        encoding="utf-8-sig",
    )

    return combined_df


def create_manifest_for_opportunity(
    input_dir: Path,
    opportunity_id: str,
    database_target: str = "rfp_db",
    output_path: Path = OUTPUT_PATH,
    skipped_path: Path = SKIPPED_PATH,
    merge_existing: bool = True,
) -> list[dict[str, Any]]:
    """
    Create manifest rows for one uploaded RFP opportunity only.

    Existing rows for the same opportunity are replaced when
    merge_existing=True. Rows for other opportunities and proposal
    knowledge are preserved.
    """
    if database_target != "rfp_db":
        raise ValueError(
            "create_manifest_for_opportunity only supports "
            "database_target='rfp_db'."
        )

    normalized_opportunity_id = normalize_id(
        opportunity_id
    )

    if not normalized_opportunity_id:
        raise ValueError(
            "Opportunity ID cannot be empty."
        )

    input_dir = Path(input_dir)
    raw_root = RAW_ROOT.resolve()
    resolved_input_dir = input_dir.resolve()

    if not resolved_input_dir.exists():
        raise FileNotFoundError(
            f"Opportunity folder not found: {resolved_input_dir}"
        )

    if not resolved_input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a folder: {resolved_input_dir}"
        )

    try:
        resolved_input_dir.relative_to(raw_root)
    except ValueError as error:
        raise ValueError(
            "Opportunity folder must be located under "
            f"{raw_root}"
        ) from error

    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    files = sorted(
        path
        for path in resolved_input_dir.rglob("*")
        if path.is_file()
    )

    for file_path in files:
        suffix = file_path.suffix.lower()
        relative_path = file_path.relative_to(raw_root)

        if suffix not in SUPPORTED_EXTENSIONS:
            skipped_rows.append(
                {
                    "opportunity_id": (
                        normalized_opportunity_id
                    ),
                    "relative_path": (
                        relative_path.as_posix()
                    ),
                    "source_file": file_path.name,
                    "file_type": suffix.replace(".", ""),
                    "reason": "Unsupported file extension",
                }
            )
            continue

        rows.append(
            _build_manifest_row(
                file_path=file_path,
                raw_root=raw_root,
                database_target="rfp_db",
                collection_name="rfp_documents",
                folder_name=resolved_input_dir.name,
                opportunity_id=(
                    normalized_opportunity_id
                ),
                project_id=None,
                project_name=None,
            )
        )

    if not rows:
        raise ValueError(
            "No supported RFP files were found in "
            f"{resolved_input_dir}."
        )

    _save_manifest_rows(
        rows=rows,
        output_path=Path(output_path),
        opportunity_id=normalized_opportunity_id,
        merge_existing=merge_existing,
    )

    _save_skipped_rows(
        rows=skipped_rows,
        skipped_path=Path(skipped_path),
        opportunity_id=normalized_opportunity_id,
        merge_existing=merge_existing,
    )

    print(f"Updated manifest: {output_path}")
    print(
        "Opportunity ID:",
        normalized_opportunity_id,
    )
    print("Files included:", len(rows))
    print("Files skipped:", len(skipped_rows))

    return rows


def create_full_manifest(
    raw_root: Path = RAW_ROOT,
    output_path: Path = OUTPUT_PATH,
    skipped_path: Path = SKIPPED_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preserve the original behavior: scan all supported files under
    data/raw and rebuild the complete manifest.
    """
    raw_root = Path(raw_root)
    output_path = Path(output_path)
    skipped_path = Path(skipped_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw folder not found: {raw_root.resolve()}"
        )

    resolved_raw_root = raw_root.resolve()
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    files = sorted(
        path
        for path in resolved_raw_root.rglob("*")
        if path.is_file()
    )

    for file_path in files:
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(
            resolved_raw_root
        )
        relative_parts = relative_path.parts

        database_target, collection_name = (
            detect_database_target(relative_parts)
        )

        if database_target == "unknown":
            skipped_rows.append(
                {
                    "opportunity_id": "",
                    "relative_path": (
                        relative_path.as_posix()
                    ),
                    "source_file": file_path.name,
                    "file_type": suffix.replace(".", ""),
                    "reason": (
                        "File is not under "
                        "data/raw/rfp_uploads or "
                        "data/raw/proposal_knowledge"
                    ),
                }
            )
            continue

        folder_name = get_project_or_opportunity_name(
            relative_parts
        )

        if database_target == "rfp_db":
            opportunity_id = normalize_id(folder_name)
            project_id = None
            project_name = None
        else:
            opportunity_id = None
            project_id = normalize_id(folder_name)
            project_name = folder_name

        rows.append(
            _build_manifest_row(
                file_path=file_path,
                raw_root=resolved_raw_root,
                database_target=database_target,
                collection_name=collection_name,
                folder_name=folder_name,
                opportunity_id=opportunity_id,
                project_id=project_id,
                project_name=project_name,
            )
        )

    manifest_df = _save_manifest_rows(
        rows=rows,
        output_path=output_path,
        merge_existing=False,
    )

    skipped_df = _save_skipped_rows(
        rows=skipped_rows,
        skipped_path=skipped_path,
        merge_existing=False,
    )

    return manifest_df, skipped_df


def print_full_manifest_summary(
    manifest_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
) -> None:
    print(f"Created manifest: {OUTPUT_PATH}")
    print(f"Files included: {len(manifest_df)}")
    print(
        "Files skipped as unknown:",
        len(skipped_df),
    )

    if len(manifest_df) > 0:
        print("\nDatabase targets:")
        print(
            manifest_df["database_target"]
            .value_counts(dropna=False)
        )

        print("\nCollections:")
        print(
            manifest_df["collection_name"]
            .value_counts(dropna=False)
        )

        print("\nFile types:")
        print(
            manifest_df["file_type"]
            .value_counts(dropna=False)
        )

        print("\nSample:")
        print(
            manifest_df[
                [
                    "database_target",
                    "collection_name",
                    "folder_name",
                    "source_file",
                    "file_type",
                ]
            ].head(20)
        )

    if len(skipped_df) > 0:
        print("\nSkipped unknown files saved to:")
        print(SKIPPED_PATH)
        print("\nFirst skipped files:")
        print(skipped_df.head(20))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Optional uploaded opportunity folder. "
            "When omitted, the complete data/raw tree is scanned."
        ),
    )

    parser.add_argument(
        "--opportunity-id",
        default=None,
        help=(
            "Required when --input-dir is provided."
        ),
    )

    parser.add_argument(
        "--replace-manifest",
        action="store_true",
        help=(
            "For opportunity mode, replace the whole manifest "
            "instead of merging with existing rows."
        ),
    )

    args = parser.parse_args()

    if args.input_dir:
        if not args.opportunity_id:
            parser.error(
                "--opportunity-id is required when "
                "--input-dir is provided."
            )

        create_manifest_for_opportunity(
            input_dir=Path(args.input_dir),
            opportunity_id=args.opportunity_id,
            merge_existing=(
                not args.replace_manifest
            ),
        )
        return

    manifest_df, skipped_df = create_full_manifest()
    print_full_manifest_summary(
        manifest_df=manifest_df,
        skipped_df=skipped_df,
    )


if __name__ == "__main__":
    main()
