"""Export rendered Markdown proposals to PDF."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import markdown as markdown_lib
from xhtml2pdf import pisa


def markdown_to_html(markdown_text: str) -> str:
    """Convert proposal Markdown into styled HTML for PDF export."""

    body_html = markdown_lib.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
        ],
    )

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        font-family: Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.5;
        color: #222;
    }}

    h1 {{
        font-size: 22pt;
        margin-bottom: 16px;
    }}

    h2 {{
        font-size: 15pt;
        margin-top: 24px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 4px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        font-size: 8pt;
    }}

    th, td {{
        border: 1px solid #ccc;
        padding: 5px;
        vertical-align: top;
    }}

    th {{
        background-color: #f2f2f2;
    }}

    code {{
        font-family: Courier, monospace;
        font-size: 8pt;
    }}

    @page {{
        size: A4;
        margin: 18mm;
    }}
</style>
</head>
<body>
{body_html}
</body>
</html>
""".strip()


def export_markdown_to_pdf(
    markdown_text: str,
    output_path: Path,
) -> bytes:
    """
    Convert rendered Markdown into a PDF file.

    Returns the PDF bytes so Streamlit can use them directly
    in st.download_button.
    """

    if not markdown_text.strip():
        raise ValueError("Markdown content is empty.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    html = markdown_to_html(
        markdown_text
    )

    pdf_buffer = BytesIO()

    result = pisa.CreatePDF(
        src=html,
        dest=pdf_buffer,
        encoding="utf-8",
    )

    if result.err:
        raise RuntimeError(
            "PDF generation failed using xhtml2pdf."
        )

    pdf_bytes = pdf_buffer.getvalue()

    output_path.write_bytes(
        pdf_bytes
    )

    return pdf_bytes