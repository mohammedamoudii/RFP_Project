from pathlib import Path
from datetime import datetime
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


def make_id(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:12]


def normalize_id(text: str) -> str:
    return (
        text.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def detect_database_target(relative_parts: tuple[str, ...]) -> tuple[str, str]:
    if not relative_parts:
        return "unknown", "unknown"

    top_folder = relative_parts[0].lower()

    if top_folder == "rfp_uploads":
        return "rfp_db", "rfp_documents"

    if top_folder == "proposal_knowledge":
        return "proposal_db", "proposal_knowledge"

    return "unknown", "unknown"


def get_project_or_opportunity_name(relative_parts: tuple[str, ...]) -> str:
    """
    Expected:
    data/raw/rfp_uploads/<opportunity_folder>/...
    data/raw/proposal_knowledge/<project_folder>/...
    """

    if len(relative_parts) >= 2:
        return relative_parts[1]

    return "unknown"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw folder not found: {RAW_ROOT.resolve()}")

    rows = []
    skipped_rows = []

    files = [path for path in RAW_ROOT.rglob("*") if path.is_file()]

    for file_path in files:
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(RAW_ROOT)
        relative_parts = relative_path.parts

        database_target, collection_name = detect_database_target(relative_parts)

        if database_target == "unknown":
            skipped_rows.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "source_file": file_path.name,
                    "file_type": suffix.replace(".", ""),
                    "reason": "File is not under data/raw/rfp_uploads or data/raw/proposal_knowledge",
                }
            )
            continue

        folder_name = get_project_or_opportunity_name(relative_parts)
        relative_path_str = relative_path.as_posix()
        file_id = make_id(relative_path_str)
        stat = file_path.stat()

        if database_target == "rfp_db":
            opportunity_id = normalize_id(folder_name)
            project_id = None
            project_name = None

        elif database_target == "proposal_db":
            opportunity_id = None
            project_id = normalize_id(folder_name)
            project_name = folder_name

        rows.append(
            {
                "file_id": file_id,
                "database_target": database_target,
                "collection_name": collection_name,
                "opportunity_id": opportunity_id,
                "project_id": project_id,
                "project_name": project_name,
                "folder_name": folder_name,
                "source_file": file_path.name,
                "file_type": suffix.replace(".", ""),
                "relative_path": relative_path_str,
                "absolute_path": str(file_path.resolve()),
                "file_size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "status": "pending",
                "error_message": "",
            }
        )

    df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped_rows)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(SKIPPED_PATH, index=False, encoding="utf-8-sig")

    print(f"Created manifest: {OUTPUT_PATH}")
    print(f"Files included: {len(df)}")
    print(f"Files skipped as unknown: {len(skipped_df)}")

    if len(df) > 0:
        print("\nDatabase targets:")
        print(df["database_target"].value_counts(dropna=False))

        print("\nCollections:")
        print(df["collection_name"].value_counts(dropna=False))

        print("\nFile types:")
        print(df["file_type"].value_counts(dropna=False))

        print("\nSample:")
        print(df[["database_target", "collection_name", "folder_name", "source_file", "file_type"]].head(20))

    if len(skipped_df) > 0:
        print("\nSkipped unknown files saved to:")
        print(SKIPPED_PATH)
        print("\nFirst skipped files:")
        print(skipped_df.head(20))


if __name__ == "__main__":
    main()