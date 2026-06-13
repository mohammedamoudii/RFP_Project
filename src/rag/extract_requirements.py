from pathlib import Path
import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_INPUT_PATH = Path("data/processed/retrieved_rfp_context.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/extracted_requirements.json")
DEFAULT_CONTEXT_MAX_CHARS = 50000


REQUIREMENTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "opportunity_id": {
            "type": "string",
            "description": "The RFP opportunity identifier."
        },
        "source_summary": {
            "type": "string",
            "description": "Short summary of the retrieved RFP context used for extraction."
        },
        "requirements": {
            "type": "array",
            "description": "List of extracted RFP requirements.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "requirement_id": {
                        "type": "string",
                        "description": "Sequential requirement ID such as REQ-001."
                    },
                    "requirement_text": {
                        "type": "string",
                        "description": "Clear and complete requirement statement."
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
                            "other"
                        ]
                    },
                    "proposal_type": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["business", "technical"]
                        },
                        "description": "Whether this requirement supports business proposal, technical proposal, or both."
                    },
                    "priority": {
                        "type": "string",
                        "enum": [
                            "mandatory",
                            "optional",
                            "nice_to_have",
                            "unknown"
                        ]
                    },
                    "evidence_quote": {
                        "type": "string",
                        "description": "Short supporting quote or evidence from retrieved RFP context."
                    },
                    "source_file": {
                        "type": "string",
                        "description": "Source file where the requirement was found."
                    },
                    "page_number": {
                        "type": "string",
                        "description": "Page number or page range if available. Empty string if unavailable."
                    },
                    "section": {
                        "type": "string",
                        "description": "Section, sheet, or heading if available. Empty string if unavailable."
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": "Chunk ID used as supporting evidence."
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    }
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
                    "confidence"
                ]
            }
        }
    },
    "required": [
        "opportunity_id",
        "source_summary",
        "requirements"
    ]
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def infer_opportunity_id(chunks: list[dict[str, Any]]) -> str:
    for chunk in chunks:
        opportunity_id = chunk.get("opportunity_id")
        if opportunity_id:
            return str(opportunity_id)

    return "unknown"


def format_chunk_for_prompt(chunk: dict[str, Any], index: int) -> str:
    source_file = chunk.get("source_file", "")
    page_numbers = chunk.get("page_numbers", "")
    page_start = chunk.get("page_number_start", "")
    page_end = chunk.get("page_number_end", "")
    sheet_names = chunk.get("sheet_names", "")
    sections = chunk.get("sections", "")
    chunk_id = chunk.get("chunk_id", "")
    content = chunk.get("content", "")

    location_parts = []

    if page_numbers:
        location_parts.append(f"pages={page_numbers}")
    elif page_start:
        location_parts.append(f"page_start={page_start}")
    elif page_end:
        location_parts.append(f"page_end={page_end}")

    if sheet_names:
        location_parts.append(f"sheets={sheet_names}")

    if sections:
        location_parts.append(f"sections={sections}")

    location = "; ".join(location_parts) if location_parts else "location unavailable"

    return f"""
[CONTEXT ITEM {index}]
actual_chunk_id: {chunk_id}
source_file: {source_file}
location: {location}

content:
{content}
""".strip()


def build_context(chunks: list[dict[str, Any]], max_chars: int) -> str:
    parts = []
    total_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        text = format_chunk_for_prompt(chunk, index)

        if total_chars + len(text) > max_chars:
            break

        parts.append(text)
        total_chars += len(text)

    separator = "\n\n" + ("=" * 80) + "\n\n"
    return separator.join(parts)


def build_messages(opportunity_id: str, context: str):
    system_message = """
You are an RFP requirement extraction assistant.

Your job is to extract structured requirements from retrieved RFP context.

Use only the provided RFP context.
Do not invent information.
Do not use outside knowledge.
Do not use old proposal knowledge.
Return only requirements supported by the retrieved RFP chunks.
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
- Include submission, technical, compliance, security, delivery, pricing, evaluation, legal, and business requirements when present.
- Classify each requirement using the allowed category list.
- Use proposal_type = ["business"] when the requirement mainly affects business/delivery/value/pricing/timeline.
- Use proposal_type = ["technical"] when the requirement mainly affects solution design/architecture/security/data/integration/testing.
- Use proposal_type = ["business", "technical"] when both are relevant.
- Use source_file, page_number, section, and chunk_id from the supporting chunk metadata.
- For chunk_id, copy the exact value after "actual_chunk_id:".
- Never create chunk IDs like "chunk_1", "chunk_2", or "CHUNK 1".
- The chunk_id must look like the real retrieved chunk ID, for example: chunk_e84c5ba3d8a8f14e.
- If page number is unavailable, use an empty string.
- If section is unavailable, use an empty string.
- If uncertain, set confidence to low.
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def extract_requirements_with_openai(
    opportunity_id: str,
    context: str,
    model: str,
) -> dict[str, Any]:
    client = OpenAI()

    messages = build_messages(
        opportunity_id=opportunity_id,
        context=context,
    )

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rfp_requirements",
                "strict": True,
                "schema": REQUIREMENTS_SCHEMA,
            },
        },
    )

    content = completion.choices[0].message.content

    if not content:
        raise ValueError("OpenAI returned empty content.")

    return json.loads(content)


def print_summary(result: dict[str, Any]):
    requirements = result.get("requirements", [])

    print("\nExtraction complete.")
    print(f"Opportunity ID: {result.get('opportunity_id')}")
    print(f"Requirements extracted: {len(requirements)}")

    category_counts = {}
    priority_counts = {}
    confidence_counts = {}

    for req in requirements:
        category = req.get("category", "unknown")
        priority = req.get("priority", "unknown")
        confidence = req.get("confidence", "unknown")

        category_counts[category] = category_counts.get(category, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

    print("\nCategory counts:")
    for key, value in sorted(category_counts.items()):
        print(f"- {key}: {value}")

    print("\nPriority counts:")
    for key, value in sorted(priority_counts.items()):
        print(f"- {key}: {value}")

    print("\nConfidence counts:")
    for key, value in sorted(confidence_counts.items()):
        print(f"- {key}: {value}")

    print("\nFirst 5 requirements:")
    for req in requirements[:5]:
        print("\n" + "-" * 80)
        print(
            f"{req.get('requirement_id')} | "
            f"{req.get('category')} | "
            f"{req.get('priority')} | "
            f"{req.get('confidence')}"
        )
        print(req.get("requirement_text"))
        print(
            f"Source: {req.get('source_file')} | "
            f"Page: {req.get('page_number')} | "
            f"Chunk: {req.get('chunk_id')}"
        )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to retrieved RFP context JSON.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save extracted requirements JSON.",
    )

    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name.",
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_CONTEXT_MAX_CHARS,
        help="Maximum context characters sent to the model.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    retrieved_context = load_json(input_path)
    chunks = retrieved_context.get("chunks", [])

    if not chunks:
        raise ValueError(
            f"No chunks found in {input_path}. Run retrieve_rfp_context.py first."
        )

    opportunity_id = infer_opportunity_id(chunks)
    context = build_context(chunks, max_chars=args.max_context_chars)

    print(f"Loaded retrieved chunks: {len(chunks)}")
    print(f"Opportunity ID: {opportunity_id}")
    print(f"Context characters sent to model: {len(context)}")
    print(f"Model: {args.model}")

    result = extract_requirements_with_openai(
        opportunity_id=opportunity_id,
        context=context,
        model=args.model,
    )

    save_json(result, output_path)

    print(f"\nSaved extracted requirements to: {output_path}")
    print_summary(result)


if __name__ == "__main__":
    main()