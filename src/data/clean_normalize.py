from pathlib import Path
import json
import re
import hashlib
import pandas as pd
from tqdm import tqdm


INPUT_JSONL = Path("data/processed/parsed_elements.jsonl")
OUTPUT_JSONL = Path("data/processed/cleaned_documents.jsonl")
REPORT_CSV = Path("data/processed/cleaning_report.csv")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def normalize_bullets(text: str) -> str:
    bullet_chars = ["•", "●", "○", "▪", "▫", "–", "—"]
    for bullet in bullet_chars:
        text = text.replace(bullet, "-")
    return text


def remove_control_characters(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)


def fix_broken_hyphenation(text: str) -> str:
    """
    Example:
    implemen-
    tation

    becomes:
    implementation
    """
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_noise_lines(text: str) -> str:
    """
    Conservative cleanup only.
    Removes very simple standalone page-number lines.
    Does not remove headings or requirement numbering.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # Remove standalone page numbers like: 1, Page 1, Page 1 of 10
        if re.fullmatch(r"\d{1,4}", stripped):
            continue

        if re.fullmatch(r"page\s+\d{1,4}(\s+of\s+\d{1,4})?", stripped, flags=re.IGNORECASE):
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    text = remove_control_characters(text)
    text = fix_broken_hyphenation(text)
    text = normalize_bullets(text)
    text = remove_page_noise_lines(text)
    text = normalize_whitespace(text)

    return text


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"Parsed file not found: {INPUT_JSONL}. Run parse_documents.py first."
        )

    rows = read_jsonl(INPUT_JSONL)

    cleaned_rows = []
    report_rows = []

    for row in tqdm(rows, desc="Cleaning parsed elements"):
        original_content = row.get("content", "")
        cleaned_content = clean_text(original_content)

        original_length = len(str(original_content)) if original_content is not None else 0
        cleaned_length = len(cleaned_content)
        word_count = count_words(cleaned_content)

        keep = cleaned_length > 0 and word_count > 0

        report_rows.append(
            {
                "element_id": row.get("element_id"),
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

    write_jsonl(cleaned_rows, OUTPUT_JSONL)
    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    print(f"Cleaned documents saved to: {OUTPUT_JSONL}")
    print(f"Cleaning report saved to: {REPORT_CSV}")
    print(f"Input elements: {len(rows)}")
    print(f"Output cleaned elements: {len(cleaned_rows)}")
    print(f"Removed empty/no-word elements: {len(rows) - len(cleaned_rows)}")

    if cleaned_rows:
        df = pd.DataFrame(cleaned_rows)

        print("\nDatabase targets:")
        print(df["database_target"].value_counts(dropna=False))

        print("\nCollections:")
        print(df["collection_name"].value_counts(dropna=False))

        print("\nFile types:")
        print(df["file_type"].value_counts(dropna=False))

        print("\nElement types:")
        print(df["element_type"].value_counts(dropna=False))

        print("\nWord count summary:")
        print(df["word_count"].describe())


if __name__ == "__main__":
    main()