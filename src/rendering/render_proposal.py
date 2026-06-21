"""Render validated proposal JSON into a human-readable Markdown document."""

from pathlib import Path
import argparse
import json
from typing import Any


DEFAULT_INPUT_PATH = Path("data/processed/generated_proposal.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/generated_proposal.md")


REQUIRED_FIELDS = [
    "title",
    "proposal_type",
    "executive_summary",
    "understanding_of_requirements",
    "proposed_solution",
    "technical_approach",
    "business_value",
    "implementation_plan",
    "timeline",
    "relevant_experience",
    "risk_management",
    "compliance_matrix",
    "assumptions",
    "citations",
]


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load a generated proposal JSON file from disk."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


# def save_text(content: str, path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)

#     with path.open("w", encoding="utf-8") as file:
#         file.write(content)
def save_text(
    content: str,
    path: Path,
) -> None:
    """Write rendered Markdown atomically to avoid partial output files."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".part"
    )

    try:
        temporary_path.write_text(
            content,
            encoding="utf-8",
        )

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink(
                missing_ok=True
            )

def validate_proposal(proposal: dict[str, Any]) -> None:
    """Ensure generated proposal JSON has every section required to render."""

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if field not in proposal
    ]

    if missing_fields:
        raise ValueError(
            "Generated proposal is missing required fields: "
            + ", ".join(missing_fields)
        )


def clean_text(value: Any, fallback: str = "Not provided.") -> str:
    """Return display-safe text for optional proposal fields."""

    if value is None:
        return fallback

    text = str(value).strip()

    return text if text else fallback


def escape_table_cell(value: Any) -> str:
    """Escape Markdown table separators and collapse cell whitespace."""

    text = clean_text(value, fallback="")

    text = text.replace("|", "\\|")
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")

    return " ".join(text.split())


def render_compliance_matrix(rows: list[dict[str, Any]]) -> str:
    """Render compliance rows as a Markdown table."""

    header = (
        "| Requirement ID | Requirement | Response | Evidence |\n"
        "|---|---|---|---|"
    )

    if not rows:
        return header + "\n| — | No requirements provided | — | — |"

    rendered_rows = []

    for row in rows:
        requirement_id = escape_table_cell(
            row.get("requirement_id", "")
        )
        requirement_summary = escape_table_cell(
            row.get("requirement_summary", "")
        )
        response = escape_table_cell(
            row.get("response", "")
        )
        evidence_source = escape_table_cell(
            row.get("evidence_source", "")
        )

        rendered_rows.append(
            f"| {requirement_id} | "
            f"{requirement_summary} | "
            f"{response} | "
            f"{evidence_source} |"
        )

    return header + "\n" + "\n".join(rendered_rows)


def render_assumptions(assumptions: list[Any]) -> str:
    """Render proposal assumptions as Markdown bullets."""

    if not assumptions:
        return "- No assumptions were stated."

    return "\n".join(
        f"- {clean_text(assumption)}"
        for assumption in assumptions
    )


# def render_citations(citations: list[dict[str, Any]]) -> str:
#     if not citations:
#         return "- No citations were provided."

#     rendered = []

#     for index, citation in enumerate(citations, start=1):
#         claim = clean_text(
#             citation.get("claim"),
#             fallback="Claim not provided.",
#         )
#         source_file = clean_text(
#             citation.get("source_file"),
#             fallback="Unknown source",
#         )
#         page_number = clean_text(
#             citation.get("page_number"),
#             fallback="Not available",
#         )
#         chunk_id = clean_text(
#             citation.get("chunk_id"),
#             fallback="Not available",
#         )
#         project_name = clean_text(
#             citation.get("project_name"),
#             fallback="Not available",
#         )
#         ## new update adding evidence 
#         evidence_quote = clean_text(
#         citation.get("evidence_quote"),
#         fallback="Not provided",
# )
#         rendered.append(
#             f"{index}. **Claim:** {claim}\n"
#             f"   - **Evidence quote:** “{evidence_quote}”\n"
#             f"   - **Source file:** `{source_file}`\n"
#             f"   - **Page:** {page_number}\n"
#             f"   - **Chunk ID:** `{chunk_id}`\n"
#             f"   - **Project:** {project_name}"
            
#         )

#     return "\n\n".join(rendered)

def render_citations(
    citations: list[dict[str, Any]],
) -> str:
    """Render validated citation records with evidence metadata."""

    if not citations:
        return "- No citations were provided."

    rendered = []

    for index, citation in enumerate(
        citations,
        start=1,
    ):
        claim = clean_text(
            citation.get("claim"),
            fallback="Claim not provided.",
        )

        evidence_id = clean_text(
            citation.get("evidence_id"),
            fallback="Not available",
        )

        evidence_quote = clean_text(
            citation.get("evidence_quote"),
            fallback="Not provided",
        )

        source_file = clean_text(
            citation.get("source_file"),
            fallback="Unknown source",
        )

        page_number = clean_text(
            citation.get("page_number"),
            fallback="Not available",
        )

        chunk_id = clean_text(
            citation.get("chunk_id"),
            fallback="Not available",
        )

        project_name = clean_text(
            citation.get("project_name"),
            fallback="Not available",
        )

        rendered.append(
            f"{index}. **Claim:** {claim}\n"
            f"   - **Evidence ID:** `{evidence_id}`\n"
            f"   - **Evidence quote:** “{evidence_quote}”\n"
            f"   - **Source file:** `{source_file}`\n"
            f"   - **Page:** {page_number}\n"
            f"   - **Chunk ID:** `{chunk_id}`\n"
            f"   - **Project:** {project_name}"
        )

    return "\n\n".join(rendered)

def render_markdown(proposal: dict[str, Any]) -> str:
    """Combine proposal sections, matrix rows, and citations into Markdown."""

    title = clean_text(
        proposal.get("title"),
        fallback="Proposal Response",
    )

    proposal_type = clean_text(
        proposal.get("proposal_type"),
        fallback="Not specified",
    )

    compliance_matrix = render_compliance_matrix(
        proposal.get("compliance_matrix", [])
    )

    assumptions = render_assumptions(
        proposal.get("assumptions", [])
    )

    citations = render_citations(
        proposal.get("citations", [])
    )

    markdown = f"""# {title}

**Proposal Type:** {proposal_type.title()}

## 1. Executive Summary

{clean_text(proposal.get("executive_summary"))}

## 2. Understanding of Requirements

{clean_text(proposal.get("understanding_of_requirements"))}

## 3. Proposed Solution

{clean_text(proposal.get("proposed_solution"))}

## 4. Technical Approach

{clean_text(proposal.get("technical_approach"))}

## 5. Business Value

{clean_text(proposal.get("business_value"))}

## 6. Implementation Plan

{clean_text(proposal.get("implementation_plan"))}

## 7. Timeline

{clean_text(proposal.get("timeline"))}

## 8. Relevant Experience

{clean_text(proposal.get("relevant_experience"))}

## 9. Risk Management

{clean_text(proposal.get("risk_management"))}

## 10. Compliance Matrix

{compliance_matrix}

## 11. Assumptions

{assumptions}

## 12. Sources and Citations

{citations}
"""

    return markdown.strip() + "\n"


def main() -> None:
    """CLI entry point for rendering a generated proposal JSON file."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to generated proposal JSON.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save rendered Markdown.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    proposal = load_json(input_path)
    validate_proposal(proposal)

    markdown = render_markdown(proposal)
    save_text(markdown, output_path)

    print(f"Input proposal: {input_path}")
    print(f"Markdown saved to: {output_path}")
    print(
        "Compliance rows:",
        len(proposal.get("compliance_matrix", [])),
    )
    print(
        "Citations:",
        len(proposal.get("citations", [])),
    )
    print(
        "Markdown characters:",
        len(markdown),
    )


if __name__ == "__main__":
    main()
