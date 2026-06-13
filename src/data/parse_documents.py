from pathlib import Path
import json
import traceback
import pandas as pd
from tqdm import tqdm

from pypdf import PdfReader
from docx import Document
from pptx import Presentation


MANIFEST_PATH = Path("data/processed/file_manifest.csv")
OUTPUT_JSONL = Path("data/processed/parsed_elements.jsonl")
ERRORS_CSV = Path("data/processed/parsing_errors.csv")


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def write_jsonl(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_metadata(row):
    return {
        "file_id": row.get("file_id"),
        "database_target": row.get("database_target"),
        "collection_name": row.get("collection_name"),
        "opportunity_id": row.get("opportunity_id") if pd.notna(row.get("opportunity_id")) else None,
        "project_id": row.get("project_id") if pd.notna(row.get("project_id")) else None,
        "project_name": row.get("project_name") if pd.notna(row.get("project_name")) else None,
        "folder_name": row.get("folder_name"),
        "source_file": row.get("source_file"),
        "file_type": row.get("file_type"),
        "relative_path": row.get("relative_path"),
        "absolute_path": row.get("absolute_path"),
    }


def parse_pdf(file_path, row):
    elements = []
    metadata = base_metadata(row)

    reader = PdfReader(str(file_path))

    for page_index, page in enumerate(reader.pages, start=1):
        text = safe_text(page.extract_text())

        if not text:
            continue

        elements.append({
            **metadata,
            "element_id": f"{metadata['file_id']}_page_{page_index}",
            "element_type": "pdf_page",
            "page_number": page_index,
            "slide_number": None,
            "sheet_name": None,
            "row_start": None,
            "row_end": None,
            "section": None,
            "content": text,
        })

    return elements


def parse_docx(file_path, row):
    elements = []
    metadata = base_metadata(row)

    doc = Document(str(file_path))
    element_index = 0

    for para in doc.paragraphs:
        text = safe_text(para.text)

        if not text:
            continue

        element_index += 1
        elements.append({
            **metadata,
            "element_id": f"{metadata['file_id']}_paragraph_{element_index}",
            "element_type": "docx_paragraph",
            "page_number": None,
            "slide_number": None,
            "sheet_name": None,
            "row_start": None,
            "row_end": None,
            "section": None,
            "content": text,
        })

    for table_index, table in enumerate(doc.tables, start=1):
        rows_text = []

        for table_row in table.rows:
            cells = [safe_text(cell.text).replace("\n", " ") for cell in table_row.cells]
            if any(cells):
                rows_text.append(" | ".join(cells))

        if rows_text:
            elements.append({
                **metadata,
                "element_id": f"{metadata['file_id']}_table_{table_index}",
                "element_type": "docx_table",
                "page_number": None,
                "slide_number": None,
                "sheet_name": None,
                "row_start": None,
                "row_end": None,
                "section": None,
                "content": "\n".join(rows_text),
            })

    return elements


def parse_pptx(file_path, row):
    elements = []
    metadata = base_metadata(row)

    prs = Presentation(str(file_path))

    for slide_index, slide in enumerate(prs.slides, start=1):
        slide_text_parts = []

        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = safe_text(shape.text)
                if text:
                    slide_text_parts.append(text)

            if getattr(shape, "has_table", False):
                table_rows = []
                for table_row in shape.table.rows:
                    cells = [safe_text(cell.text).replace("\n", " ") for cell in table_row.cells]
                    if any(cells):
                        table_rows.append(" | ".join(cells))

                if table_rows:
                    slide_text_parts.append("\n".join(table_rows))

        slide_text = "\n\n".join(slide_text_parts).strip()

        if slide_text:
            elements.append({
                **metadata,
                "element_id": f"{metadata['file_id']}_slide_{slide_index}",
                "element_type": "pptx_slide",
                "page_number": None,
                "slide_number": slide_index,
                "sheet_name": None,
                "row_start": None,
                "row_end": None,
                "section": None,
                "content": slide_text,
            })

    return elements


def parse_text_file(file_path, row):
    metadata = base_metadata(row)

    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return []

    return [{
        **metadata,
        "element_id": f"{metadata['file_id']}_text_1",
        "element_type": "text_file",
        "page_number": None,
        "slide_number": None,
        "sheet_name": None,
        "row_start": None,
        "row_end": None,
        "section": None,
        "content": text,
    }]


def dataframe_to_text(df):
    df = df.dropna(how="all").dropna(axis=1, how="all")

    if df.empty:
        return ""

    df = df.fillna("")
    lines = []

    for _, row in df.iterrows():
        values = [safe_text(value).replace("\n", " ") for value in row.tolist()]
        if any(values):
            lines.append(" | ".join(values))

    return "\n".join(lines).strip()


def parse_xlsx(file_path, row, rows_per_element=40):
    elements = []
    metadata = base_metadata(row)

    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            header=None,
            dtype=str
        )

        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.empty:
            continue

        total_rows = len(df)

        for start in range(0, total_rows, rows_per_element):
            end = min(start + rows_per_element, total_rows)
            chunk_df = df.iloc[start:end]
            text = dataframe_to_text(chunk_df)

            if not text:
                continue

            elements.append({
                **metadata,
                "element_id": f"{metadata['file_id']}_sheet_{sheet_name}_rows_{start + 1}_{end}",
                "element_type": "xlsx_sheet_rows",
                "page_number": None,
                "slide_number": None,
                "sheet_name": sheet_name,
                "row_start": start + 1,
                "row_end": end,
                "section": sheet_name,
                "content": text,
            })

    return elements


def parse_file(row):
    file_type = row["file_type"]
    file_path = Path(row["absolute_path"])

    if file_type == "pdf":
        return parse_pdf(file_path, row)

    if file_type == "docx":
        return parse_docx(file_path, row)

    if file_type == "pptx":
        return parse_pptx(file_path, row)

    if file_type in {"md", "txt"}:
        return parse_text_file(file_path, row)

    if file_type == "xlsx":
        return parse_xlsx(file_path, row)

    return []


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. Run create_manifest.py first."
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    all_elements = []
    errors = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Parsing documents"):
        try:
            elements = parse_file(row)
            all_elements.extend(elements)

        except Exception as e:
            errors.append({
                "file_id": row.get("file_id"),
                "source_file": row.get("source_file"),
                "relative_path": row.get("relative_path"),
                "file_type": row.get("file_type"),
                "database_target": row.get("database_target"),
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            })

    write_jsonl(all_elements, OUTPUT_JSONL)

    pd.DataFrame(errors).to_csv(ERRORS_CSV, index=False, encoding="utf-8-sig")

    print(f"Parsed elements saved to: {OUTPUT_JSONL}")
    print(f"Parsing errors saved to: {ERRORS_CSV}")
    print(f"Total parsed elements: {len(all_elements)}")
    print(f"Total errors: {len(errors)}")

    if all_elements:
        preview = pd.DataFrame(all_elements)
        print("\nParsed element types:")
        print(preview["element_type"].value_counts())

        print("\nDatabase targets:")
        print(preview["database_target"].value_counts(dropna=False))

        print("\nCollections:")
        print(preview["collection_name"].value_counts(dropna=False))


if __name__ == "__main__":
    main()