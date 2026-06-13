from pathlib import Path
from collections import defaultdict
import json
import hashlib
import math
import re
import pandas as pd
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_JSONL = Path("data/processed/cleaned_documents.jsonl")
OUTPUT_JSONL = Path("data/processed/chunks.jsonl")
REPORT_CSV = Path("data/processed/chunking_report.csv")

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MIN_CHUNK_WORDS = 1


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            row["_input_order"] = idx
            rows.append(row)

    return rows


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_missing(value):
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    if str(value).strip() == "":
        return True

    return False


def normalize_scalar(value):
    if is_missing(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def count_words(text: str) -> int:
    if not text:
        return 0

    return len(re.findall(r"\b\w+\b", text))


def compact_unique(values):
    result = []

    for value in values:
        if is_missing(value):
            continue

        if isinstance(value, float) and value.is_integer():
            value = int(value)

        value = str(value).strip()

        if value and value not in result:
            result.append(value)

    return ", ".join(result) if result else None


def numeric_values(values):
    nums = []

    for value in values:
        if is_missing(value):
            continue

        try:
            nums.append(int(float(value)))
        except Exception:
            pass

    return nums


def format_element_text(row: dict) -> str:
    content = str(row.get("content", "")).strip()

    if not content:
        return ""

    element_type = row.get("element_type")
    page_number = row.get("page_number")
    slide_number = row.get("slide_number")
    sheet_name = row.get("sheet_name")
    row_start = row.get("row_start")
    row_end = row.get("row_end")
    section = row.get("section")

    prefix_parts = []

    if not is_missing(section):
        prefix_parts.append(f"Section: {section}")

    if element_type == "pdf_page" and not is_missing(page_number):
        prefix_parts.append(f"Page: {page_number}")

    if element_type == "pptx_slide" and not is_missing(slide_number):
        prefix_parts.append(f"Slide: {slide_number}")

    if element_type == "xlsx_sheet_rows":
        if not is_missing(sheet_name):
            prefix_parts.append(f"Sheet: {sheet_name}")

        if not is_missing(row_start) and not is_missing(row_end):
            prefix_parts.append(f"Rows: {row_start}-{row_end}")

    if prefix_parts:
        return "\n".join(prefix_parts) + "\n" + content

    return content


def build_chunk_metadata(elements, chunk_text, chunk_index):
    first = elements[0]

    page_nums = numeric_values([e.get("page_number") for e in elements])
    slide_nums = numeric_values([e.get("slide_number") for e in elements])
    row_starts = numeric_values([e.get("row_start") for e in elements])
    row_ends = numeric_values([e.get("row_end") for e in elements])

    file_id = first.get("file_id")
    chunk_id_base = f"{file_id}_{chunk_index}_{content_hash(chunk_text)}"

    return {
        "chunk_id": f"chunk_{content_hash(chunk_id_base)}",
        "chunk_index": chunk_index,
        "doc_id": file_id,
        "file_id": file_id,

        "database_target": first.get("database_target"),
        "collection_name": first.get("collection_name"),

        "opportunity_id": normalize_scalar(first.get("opportunity_id")),
        "project_id": normalize_scalar(first.get("project_id")),
        "project_name": normalize_scalar(first.get("project_name")),
        "folder_name": first.get("folder_name"),

        "source_file": first.get("source_file"),
        "file_type": first.get("file_type"),
        "relative_path": first.get("relative_path"),
        "absolute_path": first.get("absolute_path"),

        "element_ids": compact_unique([e.get("element_id") for e in elements]),
        "element_types": compact_unique([e.get("element_type") for e in elements]),

        "page_numbers": compact_unique([e.get("page_number") for e in elements]),
        "page_number_start": min(page_nums) if page_nums else None,
        "page_number_end": max(page_nums) if page_nums else None,

        "slide_numbers": compact_unique([e.get("slide_number") for e in elements]),
        "slide_number_start": min(slide_nums) if slide_nums else None,
        "slide_number_end": max(slide_nums) if slide_nums else None,

        "sheet_names": compact_unique([e.get("sheet_name") for e in elements]),
        "row_start": min(row_starts) if row_starts else None,
        "row_end": max(row_ends) if row_ends else None,

        "sections": compact_unique([e.get("section") for e in elements]),

        "content": chunk_text.strip(),
        "content_hash": content_hash(chunk_text),
        "chunk_char_count": len(chunk_text),
        "chunk_word_count": count_words(chunk_text),
        "is_short_chunk": count_words(chunk_text) < 20,
    }


def add_split_texts_to_chunks(split_texts, elements, file_id, chunk_counter, chunks):
    for split_text in split_texts:
        split_text = split_text.strip()

        if not split_text:
            continue

        if count_words(split_text) < MIN_CHUNK_WORDS:
            continue

        chunk_counter[file_id] += 1
        chunk_index = chunk_counter[file_id]

        chunk = build_chunk_metadata(
            elements=elements,
            chunk_text=split_text,
            chunk_index=chunk_index,
        )

        chunks.append(chunk)


def flush_buffer(buffer_elements, buffer_texts, chunk_counter, chunks):
    if not buffer_elements or not buffer_texts:
        return

    full_text = "\n\n".join(buffer_texts).strip()

    if not full_text:
        return

    if count_words(full_text) < MIN_CHUNK_WORDS:
        return

    split_texts = splitter.split_text(full_text)
    file_id = buffer_elements[0].get("file_id")

    add_split_texts_to_chunks(
        split_texts=split_texts,
        elements=buffer_elements,
        file_id=file_id,
        chunk_counter=chunk_counter,
        chunks=chunks,
    )


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"Cleaned file not found: {INPUT_JSONL}. Run clean_normalize.py first."
        )

    rows = read_jsonl(INPUT_JSONL)
    rows = sorted(rows, key=lambda r: (r.get("file_id", ""), r.get("_input_order", 0)))

    grouped = defaultdict(list)

    for row in rows:
        grouped[row.get("file_id")].append(row)

    chunks = []
    chunk_counter = defaultdict(int)

    for file_id, file_elements in tqdm(grouped.items(), desc="Creating chunks"):
        buffer_elements = []
        buffer_texts = []
        buffer_char_count = 0

        for element in file_elements:
            element_text = format_element_text(element)

            if not element_text:
                continue

            if count_words(element_text) < MIN_CHUNK_WORDS:
                continue

            element_char_count = len(element_text)

            # If one element is larger than the target chunk size,
            # flush the current buffer first, then split this large element alone.
            if element_char_count > CHUNK_SIZE:
                flush_buffer(buffer_elements, buffer_texts, chunk_counter, chunks)

                buffer_elements = []
                buffer_texts = []
                buffer_char_count = 0

                split_texts = splitter.split_text(element_text)

                add_split_texts_to_chunks(
                    split_texts=split_texts,
                    elements=[element],
                    file_id=file_id,
                    chunk_counter=chunk_counter,
                    chunks=chunks,
                )

                continue

            # If adding the next element would exceed target size,
            # flush the current buffer and start a new one.
            if buffer_texts and (buffer_char_count + element_char_count) > CHUNK_SIZE:
                flush_buffer(buffer_elements, buffer_texts, chunk_counter, chunks)

                buffer_elements = []
                buffer_texts = []
                buffer_char_count = 0

            buffer_elements.append(element)
            buffer_texts.append(element_text)
            buffer_char_count += element_char_count

        flush_buffer(buffer_elements, buffer_texts, chunk_counter, chunks)

    write_jsonl(chunks, OUTPUT_JSONL)

    report_rows = []

    for chunk in chunks:
        report_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "database_target": chunk["database_target"],
                "collection_name": chunk["collection_name"],
                "source_file": chunk["source_file"],
                "file_type": chunk["file_type"],
                "element_types": chunk["element_types"],
                "page_numbers": chunk["page_numbers"],
                "sheet_names": chunk["sheet_names"],
                "chunk_char_count": chunk["chunk_char_count"],
                "chunk_word_count": chunk["chunk_word_count"],
                "is_short_chunk": chunk["is_short_chunk"],
            }
        )

    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    print(f"Chunks saved to: {OUTPUT_JSONL}")
    print(f"Chunking report saved to: {REPORT_CSV}")
    print(f"Input cleaned elements: {len(rows)}")
    print(f"Output chunks: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Minimum chunk words: {MIN_CHUNK_WORDS}")

    if chunks:
        df = pd.DataFrame(chunks)

        print("\nDatabase targets:")
        print(df["database_target"].value_counts(dropna=False))

        print("\nCollections:")
        print(df["collection_name"].value_counts(dropna=False))

        print("\nFile types:")
        print(df["file_type"].value_counts(dropna=False))

        print("\nChunk word count summary:")
        print(df["chunk_word_count"].describe())

        print("\nShort chunks:")
        print(df["is_short_chunk"].value_counts(dropna=False))

        print("\nZero-word chunks:")
        print(len(df[df["chunk_word_count"] == 0]))


if __name__ == "__main__":
    main()