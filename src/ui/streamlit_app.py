from pathlib import Path
import json
from typing import Any

import streamlit as st


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROPOSAL_JSON_PATH = (
    PROJECT_ROOT / "data" / "processed" / "generated_proposal.json"
)

PROPOSAL_MARKDOWN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "generated_proposal.md"
)

EDITED_MARKDOWN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "generated_proposal_edited.md"
)


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="RFP Intelligence System",
    page_icon="📄",
    layout="wide",
)


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Entire application */
        .stApp,
        div[data-testid="stAppViewContainer"] {
            background-color: #f4f6f8 !important;
            color: #172033 !important;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        /* Header card */
        .app-header {
            background-color: #ffffff;
            border: 1px solid #dbe2ea;
            border-radius: 14px;
            padding: 24px 30px;
            margin-bottom: 22px;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
        }

        .app-header h1 {
            color: #172033 !important;
            margin: 0;
            font-size: 2rem;
        }

        .app-header p {
            color: #596579 !important;
            margin-top: 8px;
            margin-bottom: 0;
        }

        /* Streamlit headings */
        div[data-testid="stHeadingWithActionElements"] h1,
        div[data-testid="stHeadingWithActionElements"] h2,
        div[data-testid="stHeadingWithActionElements"] h3 {
            color: #172033 !important;
            opacity: 1 !important;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #dbe2ea !important;
            border-radius: 12px !important;
            padding: 16px !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] p {
            color: #596579 !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div {
            color: #172033 !important;
            opacity: 1 !important;
        }

        /* Tabs */
        div[data-testid="stTabs"] button {
            color: #596579 !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #4f6bed !important;
        }

        /* Proposal document container */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border-color: #dbe2ea !important;
            border-radius: 14px !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]
        div[data-testid="stMarkdownContainer"] {
            color: #293548 !important;
        }

        /* Markdown headings and paragraphs */
        div[data-testid="stMarkdownContainer"] h1 {
            color: #172033 !important;
            border-bottom: 3px solid #4f6bed;
            padding-bottom: 10px;
        }

        div[data-testid="stMarkdownContainer"] h2 {
            color: #23395d !important;
            margin-top: 2rem;
            padding-bottom: 6px;
            border-bottom: 1px solid #d8dee9;
        }

        div[data-testid="stMarkdownContainer"] h3 {
            color: #334e75 !important;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stMarkdownContainer"] strong {
            color: #293548 !important;
            opacity: 1 !important;
            line-height: 1.7;
        }

        /* Information box */
        .source-note {
            background-color: #eef6ff;
            border-left: 5px solid #4f6bed;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            color: #334155 !important;
        }

        /* Proposal tables */
        div[data-testid="stMarkdownContainer"] table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.92rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stMarkdownContainer"] table thead th {
            background-color: #dfe9fb !important;
            color: #183153 !important;
            font-weight: 700 !important;
            padding: 12px !important;
            border: 1px solid #cbd5e1 !important;
            text-align: left !important;
            vertical-align: middle !important;
        }

        div[data-testid="stMarkdownContainer"] table tbody td {
            color: #1f2937 !important;
            background-color: #ffffff !important;
            padding: 12px !important;
            border: 1px solid #dbe2ea !important;
            vertical-align: top !important;
            line-height: 1.55 !important;
            overflow-wrap: anywhere !important;
        }

        div[data-testid="stMarkdownContainer"]
        table tbody tr:nth-child(even) td {
            background-color: #f3f6fa !important;
        }

        div[data-testid="stMarkdownContainer"] table th:nth-child(1),
        div[data-testid="stMarkdownContainer"] table td:nth-child(1) {
            width: 8%;
        }

        div[data-testid="stMarkdownContainer"] table th:nth-child(2),
        div[data-testid="stMarkdownContainer"] table td:nth-child(2) {
            width: 43%;
        }

        div[data-testid="stMarkdownContainer"] table th:nth-child(3),
        div[data-testid="stMarkdownContainer"] table td:nth-child(3) {
            width: 24%;
        }

        div[data-testid="stMarkdownContainer"] table th:nth-child(4),
        div[data-testid="stMarkdownContainer"] table td:nth-child(4) {
            width: 25%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    return path.read_text(encoding="utf-8")


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def display_file_error(error: Exception) -> None:
    st.error(str(error))
    st.info(
        "Run the proposal generation and Markdown rendering scripts "
        "before opening this application."
    )
    st.stop()


# -------------------------------------------------------------------
# Load generated proposal
# -------------------------------------------------------------------

try:
    proposal = load_json(PROPOSAL_JSON_PATH)
    original_markdown = load_text(PROPOSAL_MARKDOWN_PATH)
except (FileNotFoundError, json.JSONDecodeError) as error:
    display_file_error(error)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <h1>RFP Intelligence System</h1>
        <p>
            Structured proposal preview, editing, source review,
            and Markdown export.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Proposal summary
# -------------------------------------------------------------------

title = proposal.get("title", "Proposal Response")
proposal_type = proposal.get("proposal_type", "Unknown")
compliance_rows = proposal.get("compliance_matrix", [])
citations = proposal.get("citations", [])
assumptions = proposal.get("assumptions", [])

st.subheader(title)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    label="Proposal type",
    value=str(proposal_type).title(),
)

metric_2.metric(
    label="Compliance requirements",
    value=len(compliance_rows),
)

metric_3.metric(
    label="Citations",
    value=len(citations),
)

metric_4.metric(
    label="Assumptions",
    value=len(assumptions),
)


# -------------------------------------------------------------------
# Main tabs
# -------------------------------------------------------------------

preview_tab, edit_tab, sources_tab = st.tabs(
    [
        "Styled Preview",
        "Copy / Edit",
        "Sources and Data",
    ]
)


# -------------------------------------------------------------------
# Styled preview tab
# -------------------------------------------------------------------

with preview_tab:
    st.markdown(
        """
        <div class="source-note">
            This proposal was generated from structured RFP requirements
            and retrieved proposal knowledge.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(original_markdown)

    st.download_button(
        label="Download Markdown",
        data=original_markdown,
        file_name="generated_proposal.md",
        mime="text/markdown",
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Copy / edit tab
# -------------------------------------------------------------------

with edit_tab:
    st.info(
        "The text below is editable. Select and copy it directly, "
        "or edit it before downloading."
    )

    edited_markdown = st.text_area(
        label="Editable Markdown",
        value=original_markdown,
        height=750,
        key="editable_markdown",
    )

    left_button, right_button = st.columns(2)

    with left_button:
        st.download_button(
            label="Download Edited Markdown",
            data=edited_markdown,
            file_name="generated_proposal_edited.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with right_button:
        if st.button(
            "Save Edited Copy Locally",
            use_container_width=True,
        ):
            save_text(
                path=EDITED_MARKDOWN_PATH,
                content=edited_markdown,
            )

            st.success(
                "Edited Markdown saved to "
                "data/processed/generated_proposal_edited.md"
            )


# -------------------------------------------------------------------
# Sources and data tab
# -------------------------------------------------------------------

with sources_tab:
    st.subheader("Proposal metadata")

    metadata = {
        "title": proposal.get("title"),
        "proposal_type": proposal.get("proposal_type"),
        "compliance_rows": len(compliance_rows),
        "citations": len(citations),
        "assumptions": len(assumptions),
    }

    st.json(metadata)

    st.subheader("Citations")

    if citations:
        for index, citation in enumerate(citations, start=1):
            with st.expander(
                f"Citation {index}: "
                f"{citation.get('source_file', 'Unknown source')}"
            ):
                st.markdown(
                    f"""
**Claim**

{citation.get("claim", "Not provided")}

**Source file**

`{citation.get("source_file", "Not provided")}`

**Page**

{citation.get("page_number", "Not provided")}

**Chunk ID**

`{citation.get("chunk_id", "Not provided")}`

**Project**

{citation.get("project_name", "Not provided")}
"""
                )
    else:
        st.warning("No citations were found in the generated proposal.")

    st.divider()

    st.download_button(
        label="Download Proposal JSON",
        data=json.dumps(
            proposal,
            ensure_ascii=False,
            indent=2,
        ),
        file_name="generated_proposal.json",
        mime="application/json",
        use_container_width=True,
    )