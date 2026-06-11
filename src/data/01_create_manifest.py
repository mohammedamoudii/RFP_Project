from pathlib import Path
import pandas as pd
import hashlib
from datetime import datetime

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/file_manifest.csv")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".md",
}


def make_file_id(relative_path: str) -> str:
    return hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:12]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR.resolve()}")

    files = [p for p in RAW_DIR.rglob("*") if p.is_file()]

    for file_path in files:
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(RAW_DIR).as_posix()
        parts = file_path.relative_to(RAW_DIR).parts
        if len(parts) >= 3 and parts[0] == "Tenders - Raw and Annotated":
            project_name = parts[1]
        elif len(parts) >= 2:
            project_name = parts[0]
        else:
            project_name = "unknown_project"
        project_id = (
                project_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("/", "_")
                .replace("\\", "_")
                )
            
        stat = file_path.stat()

        rows.append(
            {
                "file_id": make_file_id(relative_path),
                "project_id": project_id,
                "project_name": project_name,
                "source_file": file_path.name,
                "file_type": suffix.replace(".", ""),
                "relative_path": relative_path,
                "absolute_path": str(file_path.resolve()),
                "file_size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "status": "pending",
                "error_message": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Created manifest: {OUTPUT_PATH}")
    print(f"Files found: {len(df)}")

    if len(df) > 0:
        print(df.head())


if __name__ == "__main__":
    main()