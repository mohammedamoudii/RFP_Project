"""Extract structured RFP requirements from retrieved current-opportunity context."""

from __future__ import annotations

from pathlib import Path
import argparse
import copy
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_INPUT_PATH = Path(
    "data/processed/retrieved_rfp_context.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/processed/extracted_requirements.json"
)
DEFAULT_CONTEXT_MAX_CHARS = 50000


REQUIREMENTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "opportunity_id": {
            "type": "string",
            "description": "The RFP opportunity identifier.",
        },
        "source_summary": {
            "type": "string",
            "description": (
                "Short summary of the retrieved RFP "
                "context used for extraction."
            ),
        },
        "requirements": {
            "type": "array",
            "description": (
                "List of extracted RFP requirements."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "requirement_id": {
                        "type": "string",
                        "description": (
                            "Sequential requirement ID "
                            "such as REQ-001."
                        ),
                    },
                    "requirement_text": {
                        "type": "string",
                        "description": (
                            "Clear and complete requirement "
                            "statement."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "business",
                            "technical",
                            "compliance",
                            "security",
                            "delivery",
                            "pricing",
                            "evaluation",
                            "submission",
                            "legal",
                            "other",
                        ],
                    },
                    "proposal_type": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "business",
                                "technical",
                            ],
                        },
                        "description": (
                            "Whether this requirement supports "
                            "business proposal, technical "
                            "proposal, or both."
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "enum": [
                            "mandatory",
                            "optional",
                            "nice_to_have",
                            "unknown",
                        ],
                    },
                    "evidence_quote": {
                        "type": "string",
                        "description": (
                            "Short supporting quote or evidence "
                            "from retrieved RFP context."
                        ),
                    },
                    "source_file": {
                        "type": "string",
                        "description": (
                            "Source file where the requirement "
                            "was found."
                        ),
                    },
                    "page_number": {
                        "type": "string",
                        "description": (
                            "Page number or page range if "
                            "available. Empty if unavailable."
                        ),
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "Section, sheet, or heading if "
                            "available. Empty if unavailable."
                        ),
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": (
                            "Chunk ID used as supporting "
                            "evidence."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": [
                            "high",
                            "medium",
                            "low",
                        ],
                    },
                },
                "required": [
                    "requirement_id",
                    "requirement_text",
                    "category",
                    "proposal_type",
                    "priority",
                    "evidence_quote",
                    "source_file",
                    "page_number",
                    "section",
                    "chunk_id",
                    "confidence",
                ],
            },
        },
    },
    "required": [
        "opportunity_id",
        "source_summary",
        "requirements",
    ],
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load retrieved RFP context or extracted requirements JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Save extracted requirements JSON through a temporary file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".part"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink(
                missing_ok=True
            )


def validate_retrieved_context(
    retrieved_context: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Validate that retrieval output is scoped to one RFP opportunity."""

    opportunity_id = str(
        retrieved_context.get(
            "opportunity_id",
            "",
        )
        or ""
    ).strip()

    chunks = retrieved_context.get(
        "chunks",
        [],
    )

    if not opportunity_id:
        raise ValueError(
            "Retrieved context has no opportunity_id."
        )

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "Retrieved context contains no chunks."
        )

    wrong_scope = [
        chunk.get("chunk_id")
        for chunk in chunks
        if (
            str(
                chunk.get(
                    "opportunity_id",
                    "",
                )
                or ""
            ).strip()
            != opportunity_id
            or chunk.get(
                "database_target"
            )
            != "rfp_db"
            or chunk.get(
                "collection_name"
            )
            != "rfp_documents"
        )
    ]

    if wrong_scope:
        raise ValueError(
            "Retrieved context contains chunks outside "
            f"the active RFP scope: {wrong_scope[:5]}"
        )

    chunk_ids = [
        str(
            chunk.get(
                "chunk_id",
                "",
            )
        ).strip()
        for chunk in chunks
    ]

    if any(
        not chunk_id
        for chunk_id in chunk_ids
    ):
        raise ValueError(
            "Retrieved context contains a chunk "
            "without chunk_id."
        )

    if len(chunk_ids) != len(
        set(chunk_ids)
    ):
        raise ValueError(
            "Retrieved context contains duplicate "
            "chunk IDs."
        )

    return opportunity_id, chunks


def format_chunk_for_prompt(
    chunk: dict[str, Any],
    index: int,
) -> str:
    """Format one retrieved RFP chunk with citation metadata for the model."""

    source_file = chunk.get(
        "source_file",
        "",
    )
    page_numbers = chunk.get(
        "page_numbers",
        "",
    )
    page_start = chunk.get(
        "page_number_start",
        "",
    )
    page_end = chunk.get(
        "page_number_end",
        "",
    )
    sheet_names = chunk.get(
        "sheet_names",
        "",
    )
    sections = chunk.get(
        "sections",
        "",
    )
    chunk_id = chunk.get(
        "chunk_id",
        "",
    )
    content = chunk.get(
        "content",
        "",
    )

    location_parts = []

    if page_numbers:
        location_parts.append(
            f"pages={page_numbers}"
        )
    elif page_start:
        location_parts.append(
            f"page_start={page_start}"
        )
    elif page_end:
        location_parts.append(
            f"page_end={page_end}"
        )

    if sheet_names:
        location_parts.append(
            f"sheets={sheet_names}"
        )

    if sections:
        location_parts.append(
            f"sections={sections}"
        )

    location = (
        "; ".join(location_parts)
        if location_parts
        else "location unavailable"
    )

    return f"""
[CONTEXT ITEM {index}]
actual_chunk_id: {chunk_id}
source_file: {source_file}
location: {location}

content:
{content}
""".strip()


def build_context(
    chunks: list[dict[str, Any]],
    max_chars: int,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    """Build the bounded RFP context sent to requirement extraction."""

    if max_chars < 1:
        raise ValueError(
            "max_chars must be greater than zero."
        )

    parts = []
    included_chunks = []
    total_chars = 0

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        text = format_chunk_for_prompt(
            chunk,
            index,
        )

        if (
            total_chars + len(text)
            > max_chars
        ):
            break

        parts.append(text)
        included_chunks.append(chunk)
        total_chars += len(text)

    if not included_chunks:
        raise ValueError(
            "No retrieved chunks fit inside the "
            "configured context limit."
        )

    separator = (
        "\n\n"
        + ("=" * 80)
        + "\n\n"
    )

    return (
        separator.join(parts),
        included_chunks,
    )


def build_messages(
    opportunity_id: str,
    context: str,
) -> list[dict[str, str]]:
    """Build the system and user messages for requirement extraction."""

    system_message = """
You are an RFP requirement extraction assistant.

Extract structured requirements only from the supplied
RFP context. Do not invent information. Do not use outside
knowledge or old proposal knowledge. Every requirement must
cite one exact actual_chunk_id from the supplied context.
""".strip()

    user_message = f"""
Extract structured requirements for this RFP opportunity:

opportunity_id:
{opportunity_id}

Retrieved RFP context:
{context}

Extraction rules:
- Use only the retrieved RFP context.
- Extract explicit requirements and strongly implied requirements.
- Do not combine unrelated requirements.
- Keep each requirement clear and complete.
- Prioritize mandatory requirements when visible.
- Include submission, technical, compliance, security, delivery,
  evaluation, legal, and business requirements when present.
- Classify each requirement using the allowed category list.
- Use proposal_type = ["business"] for business, delivery, value,
  pricing, or timeline requirements.
- Use proposal_type = ["technical"] for architecture, security,
  data, integration, implementation, or testing requirements.
- Use proposal_type = ["business", "technical"] when both apply.
- Copy source_file, page_number, section, and chunk_id from the
  supporting context item.
- For chunk_id, copy the exact value after "actual_chunk_id:".
- Never invent a chunk ID.
- If page number or section is unavailable, use an empty string.
- If uncertain, use confidence = "low".
""".strip()

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def extract_requirements_with_openai(
    opportunity_id: str,
    context: str,
    model: str,
    valid_chunk_ids: list[str],
) -> dict[str, Any]:
    """Call OpenAI for structured requirement extraction."""

    requirements_schema = (
        build_requirements_schema(
            valid_chunk_ids=valid_chunk_ids
        )
    )
    client = OpenAI()

    completion = (
        client.chat.completions.create(
            model=model,
            # temperature=0,
            messages=build_messages(
                opportunity_id=(
                    opportunity_id
                ),
                context=context,
            ),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": (
                        "rfp_requirements"
                    ),
                    "strict": True,
                    "schema": (
                        requirements_schema
                    ),
                },
            },
        )
    )

    content = (
        completion
        .choices[0]
        .message
        .content
    )

    if not content:
        raise ValueError(
            "OpenAI returned empty content."
        )

    return json.loads(content)

## New 
def build_requirements_schema(
    valid_chunk_ids: list[str],
) -> dict[str, Any]:
    """Constrain extracted requirement citations to retrieved RFP chunk IDs."""

    if not valid_chunk_ids:
        raise ValueError(
            "No valid RFP chunk IDs were provided."
        )

    schema = copy.deepcopy(
        REQUIREMENTS_SCHEMA
    )

    chunk_id_property = (
        schema["properties"]
        ["requirements"]
        ["items"]
        ["properties"]
        ["chunk_id"]
    )

    chunk_id_property["enum"] = sorted(
        valid_chunk_ids
    )

    chunk_id_property["description"] = (
        "Select one exact chunk ID from the "
        "retrieved RFP context."
    )

    return schema

def extract_page_numbers(
    value: Any,
) -> set[int]:
    """Extract page numbers from model-returned page strings."""

    return {
        int(number)
        for number in re.findall(
            r"\d+",
            str(value or ""),
        )
    }


def normalize_requirement_ids(
    requirements: list[dict[str, Any]],
) -> None:
    """Assign sequential requirement IDs after validation."""

    for index, requirement in enumerate(
        requirements,
        start=1,
    ):
        requirement[
            "requirement_id"
        ] = f"REQ-{index:03d}"


def validate_extraction(
    result: dict[str, Any],
    opportunity_id: str,
    included_chunks: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """Validate extracted requirements against included RFP chunks."""

    returned_opportunity = str(
        result.get(
            "opportunity_id",
            "",
        )
        or ""
    ).strip()

    if returned_opportunity != opportunity_id:
        raise ValueError(
            "The extracted opportunity_id does not "
            "match the active opportunity."
        )

    requirements = result.get(
        "requirements",
        [],
    )

    if not isinstance(
        requirements,
        list,
    ) or not requirements:
        raise ValueError(
            "No requirements were extracted."
        )

    normalize_requirement_ids(
        requirements
    )

    chunks_by_id = {
        str(
            chunk["chunk_id"]
        ): chunk
        for chunk in included_chunks
    }

    invalid_chunk_citations = []
    source_mismatches = []
    page_warnings = []
    evidence_warnings = []

    for requirement in requirements:
        requirement_id = requirement[
            "requirement_id"
        ]

        chunk_id = str(
            requirement.get(
                "chunk_id",
                "",
            )
        ).strip()

        chunk = chunks_by_id.get(
            chunk_id
        )

        if chunk is None:
            invalid_chunk_citations.append(
                {
                    "requirement_id": (
                        requirement_id
                    ),
                    "chunk_id": chunk_id,
                }
            )
            continue

        expected_source = str(
            chunk.get(
                "source_file",
                "",
            )
            or ""
        ).strip()

        returned_source = str(
            requirement.get(
                "source_file",
                "",
            )
            or ""
        ).strip()

        if (
            expected_source
            and returned_source
            != expected_source
        ):
            source_mismatches.append(
                {
                    "requirement_id": (
                        requirement_id
                    ),
                    "expected": (
                        expected_source
                    ),
                    "returned": (
                        returned_source
                    ),
                }
            )

        returned_pages = (
            extract_page_numbers(
                requirement.get(
                    "page_number"
                )
            )
        )

        expected_pages = (
            extract_page_numbers(
                chunk.get(
                    "page_numbers"
                )
            )
        )

        if (
            returned_pages
            and expected_pages
            and returned_pages.isdisjoint(
                expected_pages
            )
        ):
            page_warnings.append(
                {
                    "requirement_id": (
                        requirement_id
                    ),
                    "expected_pages": sorted(
                        expected_pages
                    ),
                    "returned_pages": sorted(
                        returned_pages
                    ),
                }
            )

        evidence = " ".join(
            str(
                requirement.get(
                    "evidence_quote",
                    "",
                )
            ).split()
        ).lower()

        content = " ".join(
            str(
                chunk.get(
                    "content",
                    "",
                )
            ).split()
        ).lower()

        if evidence and evidence not in content:
            evidence_warnings.append(
                requirement_id
            )

    if invalid_chunk_citations:
        raise ValueError(
            "Invalid chunk citations returned by the "
            f"model: {invalid_chunk_citations[:5]}"
        )

    if source_mismatches:
        raise ValueError(
            "Source-file citations do not match their "
            f"chunks: {source_mismatches[:5]}"
        )

    requirement_ids = [
        requirement[
            "requirement_id"
        ]
        for requirement in requirements
    ]

    category_counts = {}
    priority_counts = {}
    confidence_counts = {}

    for requirement in requirements:
        category = requirement.get(
            "category",
            "unknown",
        )
        priority = requirement.get(
            "priority",
            "unknown",
        )
        confidence = requirement.get(
            "confidence",
            "unknown",
        )

        category_counts[category] = (
            category_counts.get(
                category,
                0,
            )
            + 1
        )

        priority_counts[priority] = (
            priority_counts.get(
                priority,
                0,
            )
            + 1
        )

        confidence_counts[confidence] = (
            confidence_counts.get(
                confidence,
                0,
            )
            + 1
        )

    return {
        "is_valid": True,
        "requirement_count": len(
            requirements
        ),
        "unique_requirement_ids": len(
            set(requirement_ids)
        ),
        "valid_chunk_citations": (
            len(requirements)
        ),
        "page_warnings": page_warnings,
        "evidence_quote_warnings": (
            evidence_warnings
        ),
        "category_counts": (
            category_counts
        ),
        "priority_counts": (
            priority_counts
        ),
        "confidence_counts": (
            confidence_counts
        ),
    }


def extract_requirements_from_context(
    retrieved_context: dict[str, Any],
    model: str | None = None,
    max_context_chars: int = (
        DEFAULT_CONTEXT_MAX_CHARS
    ),
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Extract, validate, and optionally save requirements from retrieval output."""

    load_dotenv(
        override=True
    )

    selected_model = (
        model
        or os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        )
    )

    opportunity_id, chunks = (
        validate_retrieved_context(
            retrieved_context
        )
    )

    context, included_chunks = (
        build_context(
            chunks=chunks,
            max_chars=max_context_chars,
        )
    )
    ## new 
    valid_chunk_ids = sorted(
        {
            str(
                chunk.get(
                    "chunk_id",
                    "",
                )
            ).strip()
            for chunk in included_chunks
            if chunk.get("chunk_id")
        }
    )

    if not valid_chunk_ids:
        raise ValueError(
            "No valid chunk IDs were included "
            "in the extraction context."
        )

    result = (extract_requirements_with_openai(
            opportunity_id=opportunity_id,
            context=context,
            model=selected_model,
            valid_chunk_ids=valid_chunk_ids,
        )
    )

    validation = validate_extraction(
        result=result,
        opportunity_id=(
            opportunity_id
        ),
        included_chunks=(
            included_chunks
        ),
    )

    result["validation"] = validation

    result["extraction_metadata"] = {
        "model": selected_model,
        "retrieved_chunk_count": len(
            chunks
        ),
        "included_chunk_count": len(
            included_chunks
        ),
        "context_character_count": len(
            context
        ),
        "max_context_characters": (
            max_context_chars
        ),
    }

    if output_path is not None:
        save_json(
            result,
            output_path,
        )

    return result


def print_summary(
    result: dict[str, Any],
) -> None:
    """Print extraction counts, warnings, and category summaries."""

    requirements = result.get(
        "requirements",
        [],
    )

    validation = result.get(
        "validation",
        {},
    )

    metadata = result.get(
        "extraction_metadata",
        {},
    )

    print("\nExtraction complete.")
    print(
        "Opportunity ID:",
        result.get(
            "opportunity_id"
        ),
    )
    print(
        "Requirements extracted:",
        len(requirements),
    )
    print(
        "Model:",
        metadata.get("model"),
    )
    print(
        "Context characters:",
        metadata.get(
            "context_character_count"
        ),
    )
    print(
        "Valid chunk citations:",
        validation.get(
            "valid_chunk_citations"
        ),
    )
    print(
        "Page warnings:",
        len(
            validation.get(
                "page_warnings",
                [],
            )
        ),
    )
    print(
        "Evidence quote warnings:",
        len(
            validation.get(
                "evidence_quote_warnings",
                [],
            )
        ),
    )

    print("\nCategory counts:")
    for key, value in sorted(
        validation.get(
            "category_counts",
            {},
        ).items()
    ):
        print(
            f"- {key}: {value}"
        )

    print("\nPriority counts:")
    for key, value in sorted(
        validation.get(
            "priority_counts",
            {},
        ).items()
    ):
        print(
            f"- {key}: {value}"
        )

    print("\nFirst 5 requirements:")

    for requirement in requirements[:5]:
        print(
            "\n" + ("-" * 80)
        )
        print(
            f"{requirement.get('requirement_id')} | "
            f"{requirement.get('category')} | "
            f"{requirement.get('priority')} | "
            f"{requirement.get('confidence')}"
        )
        print(
            requirement.get(
                "requirement_text"
            )
        )
        print(
            f"Source: "
            f"{requirement.get('source_file')} | "
            f"Page: "
            f"{requirement.get('page_number')} | "
            f"Chunk: "
            f"{requirement.get('chunk_id')}"
        )


def main() -> None:
    """CLI entry point for extracting requirements from retrieved RFP context."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=(
            DEFAULT_CONTEXT_MAX_CHARS
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    retrieved_context = load_json(
        input_path
    )

    result = (
        extract_requirements_from_context(
            retrieved_context=(
                retrieved_context
            ),
            model=args.model,
            max_context_chars=(
                args.max_context_chars
            ),
            output_path=output_path,
        )
    )

    print(
        "\nSaved extracted requirements to:",
        output_path,
    )

    print_summary(result)


if __name__ == "__main__":
    main()
