from pathlib import Path
import argparse
import csv
import json
import os
import copy
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROPOSAL_PATH = Path("data/processed/generated_proposal.json")
REQUIREMENTS_PATH = Path("data/processed/extracted_requirements.json")
PROPOSAL_CONTEXT_PATH = Path(
    "data/processed/retrieved_proposal_context.json"
)

RESULT_PATH = Path("data/evaluation/generation_results.json")
SUMMARY_CSV_PATH = Path(
    "data/evaluation/generation_summary.csv"
)
HUMAN_REVIEW_PATH = Path(
    "data/evaluation/human_generation_review.csv"
)

DEFAULT_CONTEXT_MAX_CHARS = 50000


NARRATIVE_FIELDS = [
    "executive_summary",
    "understanding_of_requirements",
    "proposed_solution",
    "technical_approach",
    "business_value",
    "implementation_plan",
    "timeline",
    "relevant_experience",
    "risk_management",
]


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "faithfulness": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                },
                "rationale": {
                    "type": "string",
                },
                "unsupported_claims": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
            },
            "required": [
                "score",
                "rationale",
                "unsupported_claims",
            ],
        },
        "answer_relevance": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                },
                "rationale": {
                    "type": "string",
                },
                "irrelevant_or_missing_points": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
            },
            "required": [
                "score",
                "rationale",
                "irrelevant_or_missing_points",
            ],
        },
        "requirement_alignment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                },
                "rationale": {
                    "type": "string",
                },
                "missing_requirement_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
            },
            "required": [
                "score",
                "rationale",
                "missing_requirement_ids",
            ],
        },
        "professional_quality": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                },
                "rationale": {
                    "type": "string",
                },
            },
            "required": [
                "score",
                "rationale",
            ],
        },
        "citation_assessment": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "citation_index": {
                        "type": "integer",
                    },
                    "chunk_id": {
                        "type": "string",
                    },
                    "supported": {
                        "type": "boolean",
                    },
                    "support_score": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],
                    },
                    "rationale": {
                        "type": "string",
                    },
                },
                "required": [
                    "citation_index",
                    "chunk_id",
                    "supported",
                    "support_score",
                    "rationale",
                ],
            },
        },
        "overall_assessment": {
            "type": "string",
        },
    },
    "required": [
        "faithfulness",
        "answer_relevance",
        "requirement_alignment",
        "professional_quality",
        "citation_assessment",
        "overall_assessment",
    ],
}

## building judge schema 
def build_judge_schema(
    expected_citation_count: int,
) -> dict[str, Any]:
    schema = copy.deepcopy(JUDGE_SCHEMA)

    citation_array = schema[
        "properties"
    ]["citation_assessment"]

    citation_array["minItems"] = (
        expected_citation_count
    )

    citation_array["maxItems"] = (
        expected_citation_count
    )

    if expected_citation_count > 0:
        citation_array[
            "items"
        ]["properties"]["citation_index"] = {
            "type": "integer",
            "enum": list(
                range(
                    1,
                    expected_citation_count + 1,
                )
            ),
        }

    return schema
## 
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows were provided for CSV output.")

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(numerator / denominator, 4)


def expected_page_from_chunk(
    chunk: dict[str, Any],
) -> str:
    page = (
        chunk.get("page_numbers")
        or chunk.get("page_number_start")
        or ""
    )

    return safe_text(page)


def build_proposal_chunk_map(
    proposal_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        safe_text(chunk.get("chunk_id")): chunk
        for chunk in proposal_context.get("chunks", [])
        if (
            chunk.get("chunk_id")
            and chunk.get("database_target") == "proposal_db"
        )
    }


def extract_chunk_ids(text: str) -> list[str]:
    return re.findall(
        r"chunk_[A-Za-z0-9]+",
        text or "",
    )


def calculate_deterministic_metrics(
    proposal: dict[str, Any],
    extracted_requirements: dict[str, Any],
    proposal_context: dict[str, Any],
) -> dict[str, Any]:
    requirements = extracted_requirements.get(
        "requirements",
        [],
    )

    compliance_rows = proposal.get(
        "compliance_matrix",
        [],
    )

    requirement_ids = {
        safe_text(req.get("requirement_id"))
        for req in requirements
        if req.get("requirement_id")
    }

    mandatory_ids = {
        safe_text(req.get("requirement_id"))
        for req in requirements
        if (
            req.get("requirement_id")
            and req.get("priority") == "mandatory"
        )
    }

    matrix_ids = {
        safe_text(row.get("requirement_id"))
        for row in compliance_rows
        if row.get("requirement_id")
    }

    covered_ids = requirement_ids.intersection(
        matrix_ids
    )

    covered_mandatory_ids = mandatory_ids.intersection(
        matrix_ids
    )

    missing_ids = sorted(
        requirement_ids.difference(matrix_ids)
    )

    unexpected_matrix_ids = sorted(
        matrix_ids.difference(requirement_ids)
    )

    narrative_completed = sum(
        1
        for field in NARRATIVE_FIELDS
        if safe_text(proposal.get(field))
    )

    chunk_map = build_proposal_chunk_map(
        proposal_context
    )

    citations = proposal.get("citations", [])

    citation_checks = []

    valid_id_count = 0
    metadata_match_count = 0
    nonempty_claim_count = 0

    for index, citation in enumerate(
        citations,
        start=1,
    ):
        chunk_id = safe_text(
            citation.get("chunk_id")
        )

        chunk = chunk_map.get(chunk_id)
        id_valid = chunk is not None

        if id_valid:
            valid_id_count += 1

        claim_nonempty = bool(
            safe_text(citation.get("claim"))
        )

        if claim_nonempty:
            nonempty_claim_count += 1

        expected_source = (
            safe_text(chunk.get("source_file"))
            if chunk
            else ""
        )

        expected_page = (
            expected_page_from_chunk(chunk)
            if chunk
            else ""
        )

        expected_project = (
            safe_text(chunk.get("project_name"))
            if chunk
            else ""
        )

        source_match = (
            id_valid
            and safe_text(
                citation.get("source_file")
            ) == expected_source
        )

        page_match = (
            id_valid
            and safe_text(
                citation.get("page_number")
            ) == expected_page
        )

        project_match = (
            id_valid
            and safe_text(
                citation.get("project_name")
            ) == expected_project
        )

        metadata_match = (
            id_valid
            and source_match
            and page_match
            and project_match
        )

        if metadata_match:
            metadata_match_count += 1

        citation_checks.append(
            {
                "citation_index": index,
                "chunk_id": chunk_id,
                "id_valid": id_valid,
                "source_match": source_match,
                "page_match": page_match,
                "project_match": project_match,
                "metadata_match": metadata_match,
                "expected_source_file": expected_source,
                "expected_page_number": expected_page,
                "expected_project_name": expected_project,
            }
        )

    compliance_chunk_ids = []

    matrix_rows_with_evidence = 0

    for row in compliance_rows:
        evidence_source = safe_text(
            row.get("evidence_source")
        )

        if (
            evidence_source
            and evidence_source.lower()
            != "evidence not found in knowledge base."
        ):
            matrix_rows_with_evidence += 1

        compliance_chunk_ids.extend(
            extract_chunk_ids(evidence_source)
        )

    valid_compliance_chunk_ids = [
        chunk_id
        for chunk_id in compliance_chunk_ids
        if chunk_id in chunk_map
    ]

    proposal_text = json.dumps(
        proposal,
        ensure_ascii=False,
    ).lower()

    unsupported_marker_count = proposal_text.count(
        "evidence not found in knowledge base"
    )

    return {
        "requirement_count": len(requirement_ids),
        "mandatory_requirement_count": len(
            mandatory_ids
        ),
        "compliance_row_count": len(
            compliance_rows
        ),
        "covered_requirement_count": len(
            covered_ids
        ),
        "requirement_coverage": ratio(
            len(covered_ids),
            len(requirement_ids),
        ),
        "mandatory_requirement_coverage": ratio(
            len(covered_mandatory_ids),
            len(mandatory_ids),
        ),
        "missing_requirement_ids": missing_ids,
        "unexpected_compliance_requirement_ids": (
            unexpected_matrix_ids
        ),
        "narrative_sections_expected": len(
            NARRATIVE_FIELDS
        ),
        "narrative_sections_completed": (
            narrative_completed
        ),
        "narrative_section_completion_rate": ratio(
            narrative_completed,
            len(NARRATIVE_FIELDS),
        ),
        "citation_count": len(citations),
        "citation_id_accuracy": ratio(
            valid_id_count,
            len(citations),
        ),
        "citation_metadata_accuracy": ratio(
            metadata_match_count,
            len(citations),
        ),
        "citation_claim_completion_rate": ratio(
            nonempty_claim_count,
            len(citations),
        ),
        "citation_checks": citation_checks,
        "compliance_rows_with_evidence": (
            matrix_rows_with_evidence
        ),
        "compliance_evidence_coverage": ratio(
            matrix_rows_with_evidence,
            len(compliance_rows),
        ),
        "compliance_chunk_references": len(
            compliance_chunk_ids
        ),
        "compliance_valid_chunk_references": len(
            valid_compliance_chunk_ids
        ),
        "compliance_chunk_id_accuracy": (
            ratio(
                len(valid_compliance_chunk_ids),
                len(compliance_chunk_ids),
            )
            if compliance_chunk_ids
            else None
        ),
        "unsupported_evidence_marker_count": (
            unsupported_marker_count
        ),
    }


def order_proposal_chunks(
    proposal: dict[str, Any],
    proposal_context: dict[str, Any],
) -> list[dict[str, Any]]:
    chunks = [
        chunk
        for chunk in proposal_context.get("chunks", [])
        if chunk.get("database_target") == "proposal_db"
    ]

    chunk_map = {
        safe_text(chunk.get("chunk_id")): chunk
        for chunk in chunks
        if chunk.get("chunk_id")
    }

    ordered = []
    seen = set()

    for citation in proposal.get("citations", []):
        chunk_id = safe_text(
            citation.get("chunk_id")
        )

        chunk = chunk_map.get(chunk_id)

        if chunk and chunk_id not in seen:
            ordered.append(chunk)
            seen.add(chunk_id)

    for chunk in chunks:
        chunk_id = safe_text(
            chunk.get("chunk_id")
        )

        if chunk_id and chunk_id not in seen:
            ordered.append(chunk)
            seen.add(chunk_id)

    return ordered


def format_chunk_for_judge(
    chunk: dict[str, Any],
    index: int,
) -> str:
    return f"""
[PROPOSAL KNOWLEDGE {index}]
actual_chunk_id: {safe_text(chunk.get("chunk_id"))}
source_file: {safe_text(chunk.get("source_file"))}
project_name: {safe_text(chunk.get("project_name"))}
page_number: {expected_page_from_chunk(chunk)}

content:
{safe_text(chunk.get("content"))}
""".strip()


def build_knowledge_context(
    proposal: dict[str, Any],
    proposal_context: dict[str, Any],
    max_chars: int,
) -> str:
    ordered_chunks = order_proposal_chunks(
        proposal=proposal,
        proposal_context=proposal_context,
    )

    parts = []
    total_chars = 0

    for index, chunk in enumerate(
        ordered_chunks,
        start=1,
    ):
        block = format_chunk_for_judge(
            chunk=chunk,
            index=index,
        )

        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars

            if remaining > 500:
                parts.append(block[:remaining])

            break

        parts.append(block)
        total_chars += len(block)

    return (
        "\n\n"
        + ("=" * 80)
        + "\n\n"
    ).join(parts)


def build_judge_messages(
    proposal: dict[str, Any],
    extracted_requirements: dict[str, Any],
    proposal_knowledge_context: str,
) -> list[dict[str, str]]:
    system_message = """
You are evaluating a retrieval-augmented RFP proposal.

Be strict and evidence-based.
Do not reward writing style when factual support is missing.

Grounding rules:
1. Client requirement statements may be supported by the extracted requirements.
2. Company capability, experience, methodology, pricing, team, implementation, and outcome claims must be supported by retrieved proposal knowledge.
3. A citation is supported only when its claim is directly supported by the exact cited chunk.
4. Do not assume facts that are not explicitly present.
5. Do not infer named methodologies such as Agile unless the retrieved text explicitly states them.
""".strip()
    citation_evidence_pairs = [
        {
            "citation_index": index,
            "claim": citation.get("claim", ""),
            "evidence_quote": citation.get(
                "evidence_quote",
                "",
            ),
            "chunk_id": citation.get(
                "chunk_id",
                "",
            ),
        }
        for index, citation in enumerate(
            proposal.get("citations", []),
            start=1,
        )
    ]
    citation_count = len(
        proposal.get("citations", [])
    )
    user_message = f"""
Evaluate the generated proposal.
There are exactly {citation_count} top-level citations in the generated proposal.

Citation evaluation rules:
- Evaluate only objects in the top-level "citations" array.
- Do not evaluate compliance_matrix evidence_source fields as citations.
- Return exactly {citation_count} citation_assessment objects.
- citation_index must correspond to the original top-level citation position, starting at 1.
EXTRACTED CLIENT REQUIREMENTS:
{json.dumps(
    extracted_requirements.get("requirements", []),
    ensure_ascii=False,
    indent=2,
)}

GENERATED PROPOSAL:
{json.dumps(
    proposal,
    ensure_ascii=False,
    indent=2,
)}

TOP-LEVEL CITATION EVIDENCE PAIRS:
{json.dumps(
    citation_evidence_pairs,
    ensure_ascii=False,
    indent=2,
)}
RETRIEVED PROPOSAL KNOWLEDGE:
{proposal_knowledge_context}

Scoring guide:

5 = fully supported or fully aligned
4 = mostly supported, with minor weaknesses
3 = partially supported
2 = major unsupported or missing content
1 = substantially unsupported or irrelevant

Tasks:
- Score faithfulness.
- Identify unsupported company claims.
- Score answer relevance.
- Score alignment with the extracted requirements.
- Identify missing requirement IDs.
- Score professional quality separately.
- Evaluate every citation in the proposal in its original order.


- For each citation, first compare the claim directly against its evidence_quote.
- The evidence_quote is an exact backend-controlled passage from the cited chunk.
- A citation is supported only when every material part of the claim is stated or clearly entailed by the evidence_quote.
- Mark the citation unsupported when the claim adds experience level, outcomes, effectiveness, success, client sectors, or capabilities not stated in the evidence_quote.
- Use the full cited chunk only to understand the quote's context, not to expand the claim beyond the quote.

- Use citation_index values starting from 1.
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


def run_llm_judge(
    proposal: dict[str, Any],
    extracted_requirements: dict[str, Any],
    proposal_knowledge_context: str,
    model: str,
) -> dict[str, Any]:
    client = OpenAI()

    messages = build_judge_messages(
        proposal=proposal,
        extracted_requirements=extracted_requirements,
        proposal_knowledge_context=(
            proposal_knowledge_context
        ),
    )
    expected_citation_count = len(
        proposal.get("citations", [])
    )

    judge_schema = build_judge_schema(
        expected_citation_count=(
            expected_citation_count
        )
    )
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "generation_evaluation",
                "strict": True,
                "schema": judge_schema,
            },
        },
    )

    message = completion.choices[0].message

    refusal = getattr(
        message,
        "refusal",
        None,
    )

    if refusal:
        raise ValueError(
            f"Evaluation model refused the request: {refusal}"
        )

    if not message.content:
        raise ValueError(
            "Evaluation model returned empty content."
        )

    return json.loads(message.content)


def normalize_five_point_score(
    score: int,
) -> float:
    return round((score - 1) / 4, 4)


def build_derived_metrics(
    judge_result: dict[str, Any],
    expected_citation_count: int,
) -> dict[str, Any]:
    raw_assessments = judge_result.get(
        "citation_assessment",
        [],
    )

    valid_assessments = []
    seen_indexes = set()
    invalid_indexes = []
    duplicate_indexes = []

    for assessment in raw_assessments:
        try:
            citation_index = int(
                assessment.get("citation_index")
            )
        except (TypeError, ValueError):
            invalid_indexes.append(
                assessment.get("citation_index")
            )
            continue

        if not (
            1
            <= citation_index
            <= expected_citation_count
        ):
            invalid_indexes.append(
                citation_index
            )
            continue

        if citation_index in seen_indexes:
            duplicate_indexes.append(
                citation_index
            )
            continue

        seen_indexes.add(citation_index)
        valid_assessments.append(
            assessment
        )

    supported_count = sum(
        1
        for item in valid_assessments
        if item.get("supported") is True
    )

    support_scores = [
        int(item.get("support_score", 1))
        for item in valid_assessments
    ]

    # Missing assessments are conservatively treated
    # as unsupported.
    citation_support_rate = ratio(
        supported_count,
        expected_citation_count,
    )
    ## new updated 
    missing_assessment_count = max(
        expected_citation_count
        - len(valid_assessments),
        0,
    )

    # A missing assessment receives the minimum score of 1.
    adjusted_support_scores = (
        support_scores
        + [1] * missing_assessment_count
    )

    mean_citation_support_score = (
        round(
            sum(adjusted_support_scores)
            / expected_citation_count,
            4,
        )
        if expected_citation_count
        else 1.0
    )

    missing_indexes = sorted(
        set(
            range(
                1,
                expected_citation_count + 1,
            )
        ).difference(seen_indexes)
    )

    faithfulness_score = int(
        judge_result["faithfulness"]["score"]
    )

    relevance_score = int(
        judge_result["answer_relevance"]["score"]
    )

    alignment_score = int(
        judge_result[
            "requirement_alignment"
        ]["score"]
    )

    normalized_scores = [
        normalize_five_point_score(
            faithfulness_score
        ),
        normalize_five_point_score(
            relevance_score
        ),
        normalize_five_point_score(
            alignment_score
        ),
        citation_support_rate,
    ]

    composite_score = round(
        sum(normalized_scores)
        / len(normalized_scores),
        4,
    )

    return {
        "faithfulness_score_1_to_5": (
            faithfulness_score
        ),
        "answer_relevance_score_1_to_5": (
            relevance_score
        ),
        "requirement_alignment_score_1_to_5": (
            alignment_score
        ),
        "professional_quality_score_1_to_5": int(
            judge_result[
                "professional_quality"
            ]["score"]
        ),
        "expected_citation_count": (
            expected_citation_count
        ),
        "raw_citation_assessment_count": len(
            raw_assessments
        ),
        "valid_citation_assessment_count": len(
            valid_assessments
        ),
        "supported_citation_count": (
            supported_count
        ),
        "citation_support_rate": (
            citation_support_rate
        ),
        "mean_citation_support_score_1_to_5": (
            mean_citation_support_score
        ),
        "missing_citation_indexes": (
            missing_indexes
        ),
        "invalid_citation_indexes": (
            invalid_indexes
        ),
        "duplicate_citation_indexes": (
            duplicate_indexes
        ),
        "generation_quality_composite_0_to_1": (
            composite_score
        ),
        "composite_note": (
            "Composite averages normalized faithfulness, "
            "relevance, requirement alignment, and citation "
            "support. Citation support uses the actual proposal "
            "citation count as its denominator."
        ),
    }

def create_human_review_template(
    path: Path,
) -> None:
    if path.exists():
        print(
            f"Human review template already exists: {path}"
        )
        return

    rows = [
        {
            "review_item": "Overall faithfulness",
            "score_1_to_5": "",
            "pass_fail": "",
            "reviewer_notes": "",
        },
        {
            "review_item": "Answer relevance",
            "score_1_to_5": "",
            "pass_fail": "",
            "reviewer_notes": "",
        },
        {
            "review_item": "Requirement coverage",
            "score_1_to_5": "",
            "pass_fail": "",
            "reviewer_notes": "",
        },
        {
            "review_item": "Citation accuracy",
            "score_1_to_5": "",
            "pass_fail": "",
            "reviewer_notes": "",
        },
        {
            "review_item": "Professional tone and clarity",
            "score_1_to_5": "",
            "pass_fail": "",
            "reviewer_notes": "",
        },
        {
            "review_item": "Compliance matrix quality",
            "score_1_to_5": "",
            "pass_fail": "",
            "reviewer_notes": "",
        },
    ]

    save_csv(rows, path)

    print(f"Human review template saved to: {path}")


def build_summary_row(
    deterministic: dict[str, Any],
    derived: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    return {
        "judge_model": model,
        "requirement_coverage": deterministic[
            "requirement_coverage"
        ],
        "mandatory_requirement_coverage": (
            deterministic[
                "mandatory_requirement_coverage"
            ]
        ),
        "narrative_section_completion_rate": (
            deterministic[
                "narrative_section_completion_rate"
            ]
        ),
        "citation_id_accuracy": deterministic[
            "citation_id_accuracy"
        ],
        "citation_metadata_accuracy": (
            deterministic[
                "citation_metadata_accuracy"
            ]
        ),
        "compliance_evidence_coverage": (
            deterministic[
                "compliance_evidence_coverage"
            ]
        ),
        "faithfulness_score_1_to_5": derived[
            "faithfulness_score_1_to_5"
        ],
        "answer_relevance_score_1_to_5": (
            derived[
                "answer_relevance_score_1_to_5"
            ]
        ),
        "requirement_alignment_score_1_to_5": (
            derived[
                "requirement_alignment_score_1_to_5"
            ]
        ),
        "professional_quality_score_1_to_5": (
            derived[
                "professional_quality_score_1_to_5"
            ]
        ),
        "citation_support_rate": derived[
            "citation_support_rate"
        ],
        "generation_quality_composite_0_to_1": (
            derived[
                "generation_quality_composite_0_to_1"
            ]
        ),
    }


def print_summary(
    deterministic: dict[str, Any],
    derived: dict[str, Any],
) -> None:
    print("\nGeneration evaluation complete.")

    print("\nDeterministic metrics:")
    print(
        "Requirement coverage:",
        deterministic["requirement_coverage"],
    )
    print(
        "Mandatory requirement coverage:",
        deterministic[
            "mandatory_requirement_coverage"
        ],
    )
    print(
        "Narrative section completion:",
        deterministic[
            "narrative_section_completion_rate"
        ],
    )
    print(
        "Citation ID accuracy:",
        deterministic[
            "citation_id_accuracy"
        ],
    )
    print(
        "Citation metadata accuracy:",
        deterministic[
            "citation_metadata_accuracy"
        ],
    )
    print(
        "Compliance evidence coverage:",
        deterministic[
            "compliance_evidence_coverage"
        ],
    )

    print("\nLLM judge metrics:")
    print(
        "Faithfulness:",
        derived["faithfulness_score_1_to_5"],
        "/ 5",
    )
    print(
        "Answer relevance:",
        derived[
            "answer_relevance_score_1_to_5"
        ],
        "/ 5",
    )
    print(
        "Requirement alignment:",
        derived[
            "requirement_alignment_score_1_to_5"
        ],
        "/ 5",
    )
    print(
        "Professional quality:",
        derived[
            "professional_quality_score_1_to_5"
        ],
        "/ 5",
    )
    print(
        "Citation support rate:",
        derived["citation_support_rate"],
    )
    print(
        "Generation quality composite:",
        derived[
            "generation_quality_composite_0_to_1"
        ],
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--proposal",
        default=str(PROPOSAL_PATH),
    )

    parser.add_argument(
        "--requirements",
        default=str(REQUIREMENTS_PATH),
    )

    parser.add_argument(
        "--proposal-context",
        default=str(PROPOSAL_CONTEXT_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(RESULT_PATH),
    )

    parser.add_argument(
        "--summary-csv",
        default=str(SUMMARY_CSV_PATH),
    )

    parser.add_argument(
        "--human-review",
        default=str(HUMAN_REVIEW_PATH),
    )

    parser.add_argument(
        "--judge-model",
        default=os.getenv(
            "OPENAI_EVAL_MODEL",
            os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini",
            ),
        ),
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_CONTEXT_MAX_CHARS,
    )

    args = parser.parse_args()

    proposal = load_json(
        Path(args.proposal)
    )

    extracted_requirements = load_json(
        Path(args.requirements)
    )

    proposal_context = load_json(
        Path(args.proposal_context)
    )

    deterministic = (
        calculate_deterministic_metrics(
            proposal=proposal,
            extracted_requirements=(
                extracted_requirements
            ),
            proposal_context=proposal_context,
        )
    )

    knowledge_context = build_knowledge_context(
        proposal=proposal,
        proposal_context=proposal_context,
        max_chars=args.max_context_chars,
    )

    print(f"Judge model: {args.judge_model}")
    print(
        "Proposal knowledge characters sent:",
        len(knowledge_context),
    )
    print(
        "Proposal citations:",
        len(proposal.get("citations", [])),
    )

    judge_result = run_llm_judge(
        proposal=proposal,
        extracted_requirements=(
            extracted_requirements
        ),
        proposal_knowledge_context=(
            knowledge_context
        ),
        model=args.judge_model,
    )

    expected_citation_count = len(
        proposal.get("citations", [])
    )

    derived = build_derived_metrics(
        judge_result=judge_result,
        expected_citation_count=(
            expected_citation_count
        ),
    )

    # warnings = []

    # expected_citation_count = len(
    #     proposal.get("citations", [])
    # )

    # judged_citation_count = len(
    #     judge_result.get(
    #         "citation_assessment",
    #         [],
    #     )
    # )

    # if judged_citation_count != expected_citation_count:
    #     warnings.append(
    #         "The LLM judge returned "
    #         f"{judged_citation_count} citation assessments "
    #         f"for {expected_citation_count} proposal citations."
    #     )
    warnings = []

    raw_judged_count = len(
        judge_result.get(
            "citation_assessment",
            [],
        )
    )

    if raw_judged_count != expected_citation_count:
        warnings.append(
            "The LLM judge returned "
            f"{raw_judged_count} citation assessments "
            f"for {expected_citation_count} proposal citations. "
            "Only valid citation indexes were used."
        )

    if derived["missing_citation_indexes"]:
        warnings.append(
            "Missing citation assessments for indexes: "
            + ", ".join(
                map(
                    str,
                    derived[
                        "missing_citation_indexes"
                    ],
                )
            )
        )

    if derived["invalid_citation_indexes"]:
        warnings.append(
            "Invalid or out-of-range citation indexes: "
            + ", ".join(
                map(
                    str,
                    derived[
                        "invalid_citation_indexes"
                    ],
                )
            )
        )

    if derived["duplicate_citation_indexes"]:
        warnings.append(
            "Duplicate citation assessment indexes: "
            + ", ".join(
                map(
                    str,
                    derived[
                        "duplicate_citation_indexes"
                    ],
                )
            )
        )
    result = {
        "evaluation_type": (
            "proposal_generation_evaluation"
        ),
        "judge_model": args.judge_model,
        "inputs": {
            "proposal": args.proposal,
            "requirements": args.requirements,
            "proposal_context": (
                args.proposal_context
            ),
        },
        "deterministic_metrics": deterministic,
        "llm_judge": judge_result,
        "derived_metrics": derived,
        "warnings": warnings,
        "limitations": [
            (
                "LLM-as-judge scores are subjective and "
                "must be supplemented with human review."
            ),
            (
                "The current proposal knowledge base contains "
                "only one historical proposal."
            ),
            (
                "The generation and judge models may be the "
                "same model, which can introduce evaluator bias."
            ),
        ],
    }

    save_json(
        result,
        Path(args.output),
    )

    summary_row = build_summary_row(
        deterministic=deterministic,
        derived=derived,
        model=args.judge_model,
    )

    save_csv(
        [summary_row],
        Path(args.summary_csv),
    )

    create_human_review_template(
        Path(args.human_review)
    )

    print_summary(
        deterministic=deterministic,
        derived=derived,
    )

    print(
        f"\nDetailed results saved to: {args.output}"
    )
    print(
        f"Summary CSV saved to: {args.summary_csv}"
    )


if __name__ == "__main__":
    main()