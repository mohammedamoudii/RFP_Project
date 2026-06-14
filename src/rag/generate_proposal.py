import hashlib
from pathlib import Path
import argparse
import copy
import json
from collections import Counter
import os
from typing import Any

import re
import unicodedata


from dotenv import load_dotenv
from openai import OpenAI


EXTRACTED_REQUIREMENTS_PATH = Path(
    "data/processed/extracted_requirements.json"
)

RETRIEVED_PROPOSAL_CONTEXT_PATH = Path(
    "data/processed/retrieved_proposal_context.json"
)

OUTPUT_PATH = Path(
    "data/processed/generated_proposal.json"
)

RAW_OUTPUT_PATH = Path(
    "data/processed/generated_proposal_raw.json"
)

EVIDENCE_CANDIDATES_PATH = Path(
    "data/processed/evidence_candidates.json"
)

APPROVED_EVIDENCE_PATH = Path(
    "data/processed/approved_evidence_ids.json"
)

DEFAULT_REQUIREMENTS_MAX_ITEMS = 30
# DEFAULT_CONTEXT_MAX_CHARS = 50000

PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {
            "type": "string"
        },
        "proposal_type": {
            "type": "string",
            "enum": ["business", "technical", "both"]
        },
        "executive_summary": {
            "type": "string"
        },
        "understanding_of_requirements": {
            "type": "string"
        },
        "proposed_solution": {
            "type": "string"
        },
        "technical_approach": {
            "type": "string"
        },
        "business_value": {
            "type": "string"
        },
        "implementation_plan": {
            "type": "string"
        },
        "timeline": {
            "type": "string"
        },
        "relevant_experience": {
            "type": "string"
        },
        "risk_management": {
            "type": "string"
        },
        "compliance_matrix": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "requirement_id": {
                        "type": "string"
                    },
                    "requirement_summary": {
                        "type": "string"
                    },
                    "response": {
                        "type": "string"
                    },
                    "evidence_source": {
                        "type": "string"
                    }
                },
                "required": [
                    "requirement_id",
                    "requirement_summary",
                    "response",
                    "evidence_source"
                ]
            }
        },
        "assumptions": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        ## updated 
"citations": {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim": {
                "type": "string"
            },
            "evidence_id": {
                "type": "string",
                "description": (
                    "A controlled evidence identifier selected from the "
                    "approved proposal knowledge evidence list."
                ),
            },
                        "source_file": {
                "type": "string"
            },
            "page_number": {
                "type": "string"
            },
            "chunk_id": {
                "type": "string"
            },
            "project_name": {
                "type": "string"
            },
        },
        "required": [
            "claim",
            "evidence_id",
            "source_file",
            "page_number",
            "chunk_id",
            "project_name"
        ],
    },
},
        
    },
    "required": [
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
        "citations"
    ]
}

# def build_proposal_schema(
#     valid_chunk_ids: list[str],
#     valid_evidence_ids: list[str],
# ) -> dict[str, Any]:
#     if not valid_chunk_ids:
#         raise ValueError(
#             "No valid proposal knowledge chunk IDs were provided."
#         )

#     if not valid_evidence_ids:
#         raise ValueError(
#             "No valid proposal evidence IDs were provided."
#         )

#     schema = copy.deepcopy(PROPOSAL_SCHEMA)

#     citation_properties = (
#         schema["properties"]
#         ["citations"]
#         ["items"]
#         ["properties"]
#     )

#     citation_properties["chunk_id"] = {
#         "type": "string",
#         "enum": sorted(valid_chunk_ids),
#         "description": (
#             "A proposal_db chunk ID. Backend metadata "
#             "will override this value."
#         ),
#     }

#     citation_properties["evidence_id"] = {
#         "type": "string",
#         "enum": sorted(valid_evidence_ids),
#         "description": (
#             "Select one exact evidence ID from the approved "
#             "proposal evidence options."
#         ),
#     }

#     return schema
def build_proposal_schema(
    valid_chunk_ids: list[str],
    valid_evidence_ids: list[str],
    valid_requirement_ids: list[str],
) -> dict[str, Any]:
    if not valid_chunk_ids:
        raise ValueError(
            "No valid proposal knowledge "
            "chunk IDs were provided."
        )

    if not valid_evidence_ids:
        raise ValueError(
            "No valid proposal evidence "
            "IDs were provided."
        )

    if not valid_requirement_ids:
        raise ValueError(
            "No valid requirement IDs "
            "were provided."
        )

    schema = copy.deepcopy(
        PROPOSAL_SCHEMA
    )

    citation_properties = (
        schema["properties"]
        ["citations"]
        ["items"]
        ["properties"]
    )

    citation_properties["chunk_id"] = {
        "type": "string",
        "enum": sorted(
            valid_chunk_ids
        ),
        "description": (
            "A proposal_db chunk ID. "
            "Backend metadata will override "
            "this value."
        ),
    }

    citation_properties["evidence_id"] = {
        "type": "string",
        "enum": sorted(
            valid_evidence_ids
        ),
        "description": (
            "Select one exact evidence ID "
            "from the approved proposal "
            "evidence options."
        ),
    }

    compliance_schema = (
        schema["properties"]
        ["compliance_matrix"]
    )

    # Require exactly one row for every
    # selected requirement.
    compliance_schema["minItems"] = len(
        valid_requirement_ids
    )

    compliance_schema["maxItems"] = len(
        valid_requirement_ids
    )

    compliance_schema[
        "items"
    ][
        "properties"
    ][
        "requirement_id"
    ] = {
        "type": "string",
        "enum": valid_requirement_ids,
        "description": (
            "Use one exact requirement ID "
            "from the selected client "
            "requirements."
        ),
    }

    return schema

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def filter_approved_evidence(
    candidate_catalog: dict[
        str,
        dict[str, Any],
    ],
    approved_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Keep only manually approved evidence items.

    On the first run, this writes all candidate evidence
    to evidence_candidates.json and stops generation so
    the evidence can be reviewed.
    """

    # Always save the latest candidate list for review.
    save_json(
        {
            "evidence": list(
                candidate_catalog.values()
            )
        },
        EVIDENCE_CANDIDATES_PATH,
    )

    if not approved_path.exists():
        raise FileNotFoundError(
            "Approved evidence file was not found. "
            f"Review {EVIDENCE_CANDIDATES_PATH} "
            f"and create {approved_path} with "
            "approved_evidence_ids."
        )

    approved_data = load_json(
        approved_path
    )

    approved_id_list = [
        str(value).strip()
        for value in approved_data.get(
            "approved_evidence_ids",
            [],
        )
        if str(value).strip()
    ]

    approved_id_counts = Counter(
        approved_id_list
    )

    duplicate_approved_ids = sorted(
        evidence_id
        for evidence_id, count
        in approved_id_counts.items()
        if count > 1
    )

    if duplicate_approved_ids:
        raise ValueError(
            "approved_evidence_ids contains duplicates: "
            + ", ".join(duplicate_approved_ids)
        )

    approved_ids = set(
        approved_id_list
    )

    if not approved_ids:
        raise ValueError(
            "approved_evidence_ids is empty. "
            "Select at least one valid evidence ID."
        )

    unknown_ids = sorted(
        approved_ids.difference(
            candidate_catalog.keys()
        )
    )

    if unknown_ids:
        raise ValueError(
            "Approved evidence contains unknown IDs: "
            + ", ".join(unknown_ids)
        )

    approved_catalog = {
        evidence_id: evidence
        for evidence_id, evidence
        in candidate_catalog.items()
        if evidence_id in approved_ids
    }

    if not approved_catalog:
        raise ValueError(
            "No approved evidence items were selected."
        )

    return approved_catalog
def format_requirement(req: dict[str, Any]) -> str:
    proposal_type = ", ".join(req.get("proposal_type", []))

    return f"""
requirement_id: {req.get("requirement_id", "")}
requirement_text: {req.get("requirement_text", "")}
category: {req.get("category", "")}
proposal_type: {proposal_type}
priority: {req.get("priority", "")}
evidence_quote: {req.get("evidence_quote", "")}
rfp_source_file: {req.get("source_file", "")}
rfp_page_number: {req.get("page_number", "")}
confidence: {req.get("confidence", "")}
""".strip()


def build_requirements_context(
    requirements: list[dict[str, Any]],
    max_items: int,
) -> str:
    selected = requirements[:max_items]
    return "\n\n" + ("-" * 80 + "\n").join(
        format_requirement(req) for req in selected
    )


# def format_proposal_chunk(chunk: dict[str, Any], index: int) -> str:
#     source_file = chunk.get("source_file", "")
#     page_numbers = chunk.get("page_numbers", "")
#     page_start = chunk.get("page_number_start", "")
#     page_end = chunk.get("page_number_end", "")
#     project_name = chunk.get("project_name", "")
#     sections = chunk.get("sections", "")
#     chunk_id = chunk.get("chunk_id", "")
#     content = chunk.get("content", "")

#     location_parts = []

#     if page_numbers:
#         location_parts.append(f"pages={page_numbers}")
#     elif page_start:
#         location_parts.append(f"page_start={page_start}")
#     elif page_end:
#         location_parts.append(f"page_end={page_end}")

#     if sections:
#         location_parts.append(f"sections={sections}")

#     location = "; ".join(location_parts) if location_parts else "location unavailable"

#     return f"""
# [PROPOSAL KNOWLEDGE ITEM {index}]
# actual_chunk_id: {chunk_id}
# source_file: {source_file}
# project_name: {project_name}
# location: {location}

# content:
# {content}
# """.strip()


# def build_proposal_knowledge_context(
#     chunks: list[dict[str, Any]],
#     max_chars: int,
# ) -> str:
#     parts = []
#     total_chars = 0

#     for index, chunk in enumerate(chunks, start=1):
#         text = format_proposal_chunk(chunk, index)

#         if total_chars + len(text) > max_chars:
#             break

#         parts.append(text)
#         total_chars += len(text)

#     separator = "\n\n" + ("=" * 80) + "\n\n"
#     return separator.join(parts)

## added new 
def build_current_opportunity_context(
    extracted: dict[str, Any],
    requirements: list[dict[str, Any]],
) -> str:
    source_files = sorted({
        str(req.get("source_file", "")).strip()
        for req in requirements
        if req.get("source_file")
    })

    current_client_name = (
        extracted.get("client_name")
        or extracted.get("organization_name")
        or extracted.get("issuer_name")
        or extracted.get("agency_name")
        or ""
    )

    return (
        f"opportunity_id: "
        f"{extracted.get('opportunity_id', '')}\n"
        f"current_client_name: "
        f"{current_client_name}\n"
        f"current_rfp_source_files: "
        f"{json.dumps(source_files, ensure_ascii=False)}"
    )
###
def build_messages(
    proposal_type: str,
    current_opportunity_context: str,
    requirements_context: str,
    evidence_context: str,
    detail_level: str,
) -> list[dict[str, str]]:
    system_message = """
You are an RFP proposal drafting assistant.

Use only:
1. Extracted current-client requirements to describe the current
   opportunity, required deliverables, and proposed commitments.
2. Approved company evidence options to describe documented past
   experience or existing company capabilities.

Do not use a current RFP requirement as proof of company experience.
Do not invent clients, sectors, experience, methodologies, tools,
team members, prices, outcomes, or capabilities.

Return JSON only using the required schema.
""".strip()

    user_message = f"""
Generate a grounded proposal draft.

proposal_type:
{proposal_type}

detail_level:
{detail_level}

CURRENT OPPORTUNITY:
{current_opportunity_context}

CLIENT REQUIREMENTS:
{requirements_context}

APPROVED COMPANY EVIDENCE OPTIONS:
{evidence_context}

GROUNDING RULES

Current opportunity and proposed commitments:
- Address all selected client requirements.
- Prioritize mandatory requirements.
- Current RFP requirements may support statements about what the
  Client requires and what the proposed solution will provide.
- Proposed actions must use future-tense wording such as:
  "we will provide", "the proposed approach will include", or
  "the project team will deliver".
- A future commitment is not proof of historical company experience.
- Do not create a top-level company citation for a future commitment,
  current deliverable, submission instruction, compliance obligation,
  timeline, training promise, documentation promise, or pricing promise.
- Do not present a historical client's requirement as an existing
  company capability.

Company evidence and top-level citations:
- Top-level citations are only for documented past experience or
  documented existing company capabilities.
- Valid citation subjects include:
  past projects, named staff experience, existing technical
  capabilities, explicitly documented methodologies, training
  experience, and documented past outcomes.
- Every citation must select exactly one evidence_id from
  APPROVED COMPANY EVIDENCE OPTIONS.
- Use each evidence_id at most once.
- Do not create multiple citation claims from the same evidence_id.
- Do not create new evidence IDs.
- A citation claim must be a short, conservative paraphrase of the
  selected exact_quote.
- Every material phrase in a citation claim must be directly supported
  by the selected exact_quote.
- Do not combine multiple capabilities into one citation claim unless
  the same exact_quote explicitly supports every capability.
- Do not add a sector, client type, business benefit, outcome,
  effectiveness claim, or project result unless it appears explicitly
  in the exact_quote.
- Do not mention the current Client, the current opportunity, or
  "the Client's needs" inside a historical company-evidence claim.
- Do not use future-tense expressions such as "will provide",
  "will deliver", "will ensure", "plan to", or "intend to" inside
  citation claims.
- Do not use strength words such as "extensive", "substantial",
  "proven", "successful", "leading", or "deep expertise" unless the
  selected exact_quote directly supports that wording.
- Fewer accurate citations are better than several broad citations.
- The citations array may be empty when no approved evidence supports
  a genuine company claim.

- Do not change a named individual's experience into a claim about the entire team or company.
- When evidence names one person, the citation claim must name that person.
- Do not add conclusions such as "demonstrating our capability", "positioning us well", or "making us well suited" unless the exact_quote explicitly states that conclusion.
- Do not write "public sector", "social services", "complex data management", or "program performance reporting" unless an approved exact_quote explicitly contains that subject.
- Executive-summary company claims must each be supported by at least one approved evidence item.
- When evidence supports only AI projects or training, do not expand it into other industries, services, methodologies, or outcomes.
- Do not use conclusions such as "demonstrating our capability",
  "showing our ability", "positioning us well", "making us well suited",
  or similar marketing conclusions.
- State only the documented person, project, organization, training
  program, or capability contained in the selected exact_quote.

Citation metadata:
- Do not write evidence_quote; the backend will attach it.
- Set chunk_id to the chunk_id shown in the selected evidence block.
- Set source_file, page_number, and project_name using values from the
  same evidence block.
- Do not invent chunk IDs, source files, project names, or page numbers.
- Do not reuse one chunk for unrelated claims.
- If page metadata is unavailable, use an empty string.

Historical-client separation:
- CURRENT OPPORTUNITY identifies the active RFP.
- Historical proposal documents are evidence sources only.
- Do not treat a historical university, agency, organization, or
  project as the current client.
  proposal recipient.
- Historical client names may appear only in relevant experience and
  only when explicitly supported by approved evidence.

- CURRENT OPPORTUNITY identifies the active RFP and current client.
- Do not treat any historical organization found in approved company
  evidence as the current client.
- Use the current_client_name value from CURRENT OPPORTUNITY.
- When current_client_name is unavailable, refer to the organization
  as "the Client".
- Historical organization names may appear only when describing
  documented past experience supported by approved evidence.


  
Narrative section requirements:

Executive summary:
- Summarize the Client need, proposed response, delivery approach,
  governance, security posture, implementation approach, and
  documented relevant experience.
- Do not merely repeat the compliance matrix.

Understanding of requirements:
- Group the requirements into logical themes such as procurement,
  functional services, security, privacy, integration, support,
  implementation, pricing, and contractual compliance.
- Explain dependencies and areas requiring confirmation.

Proposed solution:
- Describe solution components, service boundaries, expected
  deliverables, optional services, integrations, support, and
  assumptions.
- Do not claim that an unverified feature already exists.

Technical approach:
- Explain architecture, environments, identity and access,
  data handling, security validation, testing, monitoring,
  deployment, integration, and operational support.
- Mark technical details requiring vendor confirmation.

Implementation plan:
- Include phases, activities, deliverables, decision gates,
  responsible parties, dependencies, acceptance criteria,
  documentation, training, and transition to support.

Risk management:
- Include risk, cause, impact, mitigation, monitoring approach,
  owner or responsible role, and contingency action.


Compliance matrix:
- Cover every selected requirement where possible.
- requirement_summary must accurately summarize the current requirement.
- response should describe how the proposal will address the requirement.
- Do not present a proposed commitment as completed past experience.
- evidence_source may contain approved company evidence only when that
  evidence directly supports a company-capability statement in the row.
- Do not use current RFP source files or current RFP chunk IDs as
  company evidence.
- Do not place an approved evidence ID or chunk ID in evidence_source
  merely to make the row appear supported.
- When no approved company evidence directly supports the company claim,
  set evidence_source exactly to:
  "Evidence not found in knowledge base."
- Never assert that the Proponent has no litigation, judgments,
  financial instability, certifications, or regulatory findings unless
  an approved evidence quote explicitly confirms that fact.
- When confirmation is unavailable, state:
  "Requires confirmation by an authorized company representative."
- Do not invent a fixed project duration or completion date.
- When the RFP and approved evidence do not provide a duration, state
  that the detailed schedule will be confirmed during project planning,
  and record this as an assumption. 
Proposal type behavior:
- business: emphasize delivery, value, timeline, risk, and documented
  relevant experience.
- technical: emphasize solution design, architecture, security,
  integration, implementation, and testing.
- both: include both business and technical coverage.

Detail level behavior:

- concise:
  Produce a short but complete proposal.

- standard:
  Produce balanced section detail and concise compliance responses.

- detailed:
  Produce a comprehensive professional proposal without repetition,
  filler, invented facts, or unsupported claims.

  Target section lengths:
  - Executive summary: 400-600 words.
  - Understanding of requirements: 500-800 words.
  - Proposed solution: 700-1,000 words.
  - Technical approach: 900-1,400 words.
  - Business value: 500-800 words.
  - Implementation plan: 800-1,200 words.
  - Timeline: 300-500 words.
  - Relevant experience: 400-700 words, limited strictly to
    approved evidence.
  - Risk management: 600-900 words.
  - Each compliance matrix response: approximately 40-90 words.

  For every major section:
  - explain the approach;
  - identify major activities and deliverables;
  - describe controls, dependencies, and validation steps;
  - clearly distinguish documented capabilities from proposed
    future commitments;
  - state required confirmations instead of inventing information.

  Do not shorten sections merely because the compliance matrix is long.
  Do not repeat the same paragraph or marketing claim to increase length.
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
# print(f"Valid proposal citation chunk IDs: {len(valid_chunk_ids)}")
# proposal_schema = build_proposal_schema(valid_chunk_ids)
# def generate_proposal_with_openai(
#     proposal_type: str,
#     current_opportunity_context: str,
#     requirements_context: str,
#     evidence_context: str,
#     detail_level: str,
#     model: str,
#     valid_chunk_ids: list[str],
#     valid_evidence_ids: list[str],
# ) -> dict[str, Any]:
def generate_proposal_with_openai(
    proposal_type: str,
    current_opportunity_context: str,
    requirements_context: str,
    evidence_context: str,
    detail_level: str,
    model: str,
    valid_chunk_ids: list[str],
    valid_evidence_ids: list[str],
    valid_requirement_ids: list[str],
) -> dict[str, Any]:
    load_dotenv(
        dotenv_path=(
            Path(__file__)
            .resolve()
            .parents[2]
            / ".env"
        ),
        override=True,
    )

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )   
    client = OpenAI(
    api_key=api_key
    )
    ## updated messages build messages call 
    messages = build_messages(
        proposal_type=proposal_type,
        current_opportunity_context=current_opportunity_context,
        requirements_context=requirements_context,
        evidence_context=evidence_context,
        detail_level=detail_level,
    )

    # Build the schema only after valid proposal chunk IDs are available.
    # proposal_schema = build_proposal_schema(
    #     valid_chunk_ids=valid_chunk_ids,
    #     valid_evidence_ids=valid_evidence_ids,
    # )
    proposal_schema = build_proposal_schema(
        valid_chunk_ids=valid_chunk_ids,
        valid_evidence_ids=valid_evidence_ids,
        valid_requirement_ids=(
            valid_requirement_ids
        ),
    )    

    # completion = client.chat.completions.create(
    #     model=model,
    #     # temperature=0, ## reducted 
    #     messages=messages,

    completion = client.chat.completions.create(
        model=model,
        # temperature=0,

        # Encourage fuller responses.
        verbosity="high",

        # Keep enough reasoning without consuming
        # excessive completion-token capacity.
        reasoning_effort="medium",

        # Includes visible output and reasoning tokens.
        max_completion_tokens=60000,

        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rfp_proposal",
                "strict": True,
                "schema": proposal_schema,
            },
        },
    )
    finish_reason = (
        completion
        .choices[0]
        .finish_reason
    )

    print(
        "OpenAI finish reason:",
        finish_reason,
    )

    if completion.usage:
        print(
            "Prompt tokens:",
            completion.usage.prompt_tokens,
        )

        print(
            "Completion tokens:",
            completion.usage.completion_tokens,
        )

    if finish_reason == "length":
        raise ValueError(
            "The proposal reached the configured "
            "max_completion_tokens limit. Increase the "
            "limit or generate the proposal in sections."
        )
    content = completion.choices[0].message.content

    if not content:
        raise ValueError("OpenAI returned empty content.")

    return json.loads(content)
################### normalize is added new also build message() is adjusted on citation level is updated 
def normalize_match_text(value: Any) -> str:
    """
    Normalize Unicode, quotation marks, dashes, and whitespace
    without changing the actual words.
    """
    text = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    )

    replacements = {
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u00a0": " ",   # non-breaking space
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()
################# Updated refusing to save citation whose evidence quote does not actually occur 
def split_evidence_spans(
    content: Any,
    min_words: int = 5,
    max_words: int = 60,
) -> list[str]:
    """
    Divide proposal chunk content into exact reusable evidence spans.
    Every returned span comes directly from proposal_db content.
    """
    raw_text = str(content or "").strip()

    if not raw_text:
        return []

    candidates = re.split(
        r"\n+|(?<=[.!?])\s+",
        raw_text,
    )

    spans = []

    for candidate in candidates:
        candidate = " ".join(candidate.split()).strip()

        if not candidate:
            continue

        words = candidate.split()

        if len(words) < min_words:
            continue

        if len(words) <= max_words:
            spans.append(candidate)

        # Skip sentences that are too long rather than cutting
        # them into incomplete evidence fragments.
        continue
        # if len(words) <= max_words:
        #     spans.append(candidate)
        #     continue

        # # Long sentences are divided into consecutive exact passages.
        # for start in range(0, len(words), max_words):
        #     window = words[start:start + max_words]

        #     if len(window) >= min_words:
        #         spans.append(" ".join(window))

    return spans

def build_evidence_catalog(
    proposal_chunks: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    """
    Create controlled evidence IDs from proposal_db chunks.

    Invalid RFP and client-background passages are removed
    before applying the six-evidence-items-per-chunk limit.
    """

    catalog: dict[
        str,
        dict[str, Any],
    ] = {}

    seen = set()

    for chunk in proposal_chunks:
        if (
            chunk.get(
                "database_target"
            )
            != "proposal_db"
        ):
            continue

        chunk_id = str(
            chunk.get(
                "chunk_id",
                "",
            )
        ).strip()

        if not chunk_id:
            continue

        accepted_span_count = 0

        spans = split_evidence_spans(
            chunk.get(
                "content",
                "",
            )
        )

        for span in spans:
            # Filter first.
            if is_invalid_company_evidence(
                span
            ):
                continue

            normalized_span = (
                normalize_match_text(
                    span
                )
            )

            deduplication_key = (
                chunk_id,
                normalized_span,
            )

            if deduplication_key in seen:
                continue

            seen.add(
                deduplication_key
            )

            evidence_key = (
                f"{chunk_id}|"
                f"{normalized_span}"
            )

            evidence_hash = (
                hashlib.sha256(
                    evidence_key.encode(
                        "utf-8"
                    )
                )
                .hexdigest()[:12]
                .upper()
            )

            evidence_id = (
                f"EV-{evidence_hash}"
            )

            page_number = (
                chunk.get(
                    "page_numbers"
                )
                or chunk.get(
                    "page_number_start"
                )
                or ""
            )

            catalog[evidence_id] = {
                "evidence_id": (
                    evidence_id
                ),
                "quote": span,
                "chunk_id": chunk_id,
                "source_file": str(
                    chunk.get(
                        "source_file",
                        "",
                    )
                ),
                "page_number": str(
                    page_number
                ),
                "project_name": str(
                    chunk.get(
                        "project_name",
                        "",
                    )
                ),
            }

            accepted_span_count += 1

            # Limit after filtering.
            if accepted_span_count >= 6:
                break

    return catalog
# def build_evidence_catalog(
#     proposal_chunks: list[dict[str, Any]],
# ) -> dict[str, dict[str, Any]]:
#     """
#     Create controlled evidence IDs from proposal_db chunks.
#     """
#     catalog: dict[str, dict[str, Any]] = {}
#     seen = set()
#     # counter = 1

#     for chunk in proposal_chunks:
#         if chunk.get("database_target") != "proposal_db":
#             continue

#         chunk_id = str(chunk.get("chunk_id", "")).strip()

#         if not chunk_id:
#             continue

#         spans = split_evidence_spans(
#             chunk.get("content", "")
#         )[:6]

#         for span in spans:
#             if is_invalid_company_evidence(
#                 span
#             ):
#                 continue
#             normalized_span = normalize_match_text(span)
#             deduplication_key = (
#                 chunk_id,
#                 normalized_span,
#             )

#             if deduplication_key in seen:
#                 continue

#             seen.add(deduplication_key)

#             evidence_key = (
#                 f"{chunk_id}|{normalized_span}"
#             )

#             evidence_hash = hashlib.sha256(
#                 evidence_key.encode("utf-8")
#             ).hexdigest()[:12].upper()

#             evidence_id = f"EV-{evidence_hash}"

#             page_number = (
#                 chunk.get("page_numbers")
#                 or chunk.get("page_number_start")
#                 or ""
#             )

#             catalog[evidence_id] = {
#                 "evidence_id": evidence_id,
#                 "quote": span,
#                 "chunk_id": chunk_id,
#                 "source_file": str(
#                     chunk.get("source_file", "")
#                 ),
#                 "page_number": str(page_number),
#                 "project_name": str(
#                     chunk.get("project_name", "")
#                 ),
#             }

#             # counter += 1

#     return catalog

def is_invalid_company_evidence(
    quote: str,
) -> bool:
    """
    Reject passages that are not concrete company evidence.

    This includes:
    - RFP instructions
    - submission requirements
    - current-client background
    - future proposal commitments
    - generic marketing language
    - headings and document structure
    """

    text = normalize_match_text(
        quote
    )

    if not text:
        return True

    invalid_patterns = [
        r"\bproponents?\s+"
        r"(must|shall|will|are required to)\b",

        r"\brespondents?\s+"
        r"(must|shall|will|are required to)\b",

        r"\bvendors?\s+"
        r"(must|shall|will|are required to)\b",

        r"\bmust provide\b",
        r"\bmust submit\b",
        r"\bshall provide\b",
        r"\bshall submit\b",
        r"\bare required to\b",

        r"\bwith (its|their) proposal\b",
        r"\bcompleted hecvat\b",
        r"\bhecvat\b",
        r"\bsaas form\b",
        r"\bsoftware as a service form\b",
        r"\bappendix [a-z]\b",

        r"\bthe organization will\b",
        r"\bwe will\b",
        r"\bwill provide\b",
        r"\bwill deliver\b",
        r"\bwill include\b",
        r"\bwill submit\b",
        r"\bwill ensure\b",

        r"\ba table of contents\b",
        r"\btable of contents of all presented material\b",

        r"^(his|her|their|its|this|these|it)\b",

        r"\balign with usask\b",
        r"\busask['’]s (technical|operational|requirements|needs)\b",
        r"\bdirectly supporting usask\b",

        r"\bas outlined in (this|the) proposal\b",
        r"\bour organization maintains a strong commitment\b",
    ]

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in invalid_patterns
    )
def build_evidence_context(
    evidence_catalog: dict[
        str,
        dict[str, Any],
    ],
) -> str:
    """
    Validate approved evidence again before sending
    it to the proposal-generation model.
    """

    invalid_items = []

    for evidence_id, evidence in (
        evidence_catalog.items()
    ):
        quote = str(
            evidence.get(
                "quote",
                "",
            )
        ).strip()

        if is_invalid_company_evidence(
            quote
        ):
            invalid_items.append(
                f"{evidence_id}: {quote}"
            )

    if invalid_items:
        raise ValueError(
            "Approved evidence contains RFP "
            "instructions, client background, "
            "future commitments, or unsupported "
            "marketing language.\n"
            "Return to Review Evidence, retrieve "
            "the candidates again, and remove:\n- "
            + "\n- ".join(
                invalid_items
            )
        )

    blocks = []

    for evidence_id, evidence in (
        evidence_catalog.items()
    ):
        blocks.append(
            f"""
[EVIDENCE {evidence_id}]
chunk_id: {evidence["chunk_id"]}
source_file: {evidence["source_file"]}
project_name: {evidence["project_name"]}
page_number: {evidence["page_number"]}
exact_quote: {evidence["quote"]}
""".strip()
        )

    return "\n\n".join(blocks)
## new update match tokens 
# def normalize_match_tokens(value: Any) -> str:
#     """
#     Used only as a fallback for punctuation differences.
#     Word order must remain unchanged.
#     """
#     text = normalize_match_text(value)

#     text = re.sub(
#         r"[^\w\s]",
#         " ",
#         text,
#         flags=re.UNICODE,
#     )

#     text = re.sub(r"\s+", " ", text)

#     return text.strip()
####
def enrich_and_validate_citations(
    proposal: dict[str, Any],
    evidence_catalog: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    errors = []

    for index, citation in enumerate(
        proposal.get("citations", []),
        start=1,
    ):
        evidence_id = str(
            citation.get(
                "evidence_id",
                "",
            )
        ).strip()

        evidence = evidence_catalog.get(
            evidence_id
        )

        if evidence is None:
            errors.append(
                f"Citation {index} uses an unknown "
                f"evidence_id: {evidence_id}"
            )
            continue

        claim = str(
            citation.get(
                "claim",
                "",
            )
        ).strip()

        if not claim:
            errors.append(
                f"Citation {index} has an empty claim."
            )

        citation["claim"] = evidence["quote"]
        citation["evidence_quote"] = evidence["quote"]
        citation["chunk_id"] = evidence["chunk_id"]
        citation["source_file"] = evidence["source_file"]
        citation["page_number"] = evidence["page_number"]
        citation["project_name"] = evidence["project_name"]

    if errors:
        raise ValueError(
            "Invalid proposal citations:\n- "
            + "\n- ".join(errors)
        )

    return proposal
# def enrich_and_validate_citations(
#     proposal: dict[str, Any],
#     evidence_catalog: dict[str, dict[str, Any]],
# ) -> dict[str, Any]:
#     errors = []

#     for index, citation in enumerate(
#         proposal.get("citations", []),
#         start=1,
#     ):
#         evidence_id = str(
#             citation.get("evidence_id", "")
#         ).strip()

#         evidence = evidence_catalog.get(evidence_id)

#         if evidence is None:
#             errors.append(
#                 f"Citation {index} uses an unknown "
#                 f"evidence_id: {evidence_id}"
#             )
#             continue

#     claim = str(
#         citation.get("claim", "")
#     ).strip()

#     if not claim:
#         errors.append(
#             f"Citation {index} has an empty claim."
#         )

#     # Do not trust the model-written citation claim.
#     # The approved exact evidence becomes the final claim.
#     citation["claim"] = evidence["quote"]

#     # All citation evidence and metadata come from Python.
#     citation["evidence_quote"] = evidence["quote"]
#     citation["chunk_id"] = evidence["chunk_id"]
#     citation["source_file"] = evidence["source_file"]
#     citation["page_number"] = evidence["page_number"]
#     citation["project_name"] = evidence["project_name"]

#     if errors:
#         raise ValueError(
#             "Invalid proposal citations:\n- "
#             + "\n- ".join(errors)
#         )

#     return proposal
#########################
def validate_citation_claim_types(
    proposal: dict[str, Any],
) -> list[str]:
    errors = []

    future_patterns = [
        r"\bwe will\b",
        r"\bwe shall\b",
        r"\bthe proposal will\b",
        r"\bwill provide\b",
        r"\bwill deliver\b",
        r"\bwill include\b",
        r"\bwill submit\b",
        r"\bwill ensure\b",
        r"\bplan to\b",
        r"\bintend to\b",
    ]

    for index, citation in enumerate(
        proposal.get("citations", []),
        start=1,
    ):
        claim = str(
            citation.get("claim", "")
        ).casefold()

        for pattern in future_patterns:
            if re.search(pattern, claim):
                errors.append(
                    f"Citation {index} describes a future "
                    f"commitment instead of historical evidence: "
                    f"{citation.get('claim', '')}"
                )
                break

    return errors
#####
###################### this one is also updated inside valid warning there is duplicated chunk warning 
def validate_overbroad_company_claims(
    proposal: dict[str, Any],
) -> list[str]:
    errors = []

    fields = [
        "executive_summary",
        "relevant_experience",
        "business_value",
    ]

    unsupported_patterns = {
        r"\bproven methodologies\b": (
            "potentially unsupported methodology claim"
        ),
        r"\bwealth of experience\b": (
            "potentially unsupported experience-strength claim"
        ),
        r"\bpositions us well\b": (
            "unsupported suitability conclusion"
        ),
        r"\bwell positioned\b": (
            "unsupported suitability conclusion"
        ),
        r"\bwell suited\b": (
            "unsupported suitability conclusion"
        ),
        r"\bstrong track record\b": (
            "potentially unsupported track-record claim"
        ),
        r"\bdemonstrat(?:e|es|ing) our capability\b": (
            "unsupported capability conclusion"
        ),
    }

    for field in fields:
        text = str(
            proposal.get(field, "")
        ).casefold()

        for pattern, description in unsupported_patterns.items():
            if re.search(pattern, text):
                errors.append(
                    f"{field} contains unsupported claim: "
                    f"{description}"
                )

    return errors
def validate_sensitive_company_claims(
    proposal: dict[str, Any],
) -> list[str]:
    errors = []

    fields = [
        "executive_summary",
        "relevant_experience",
        "risk_management",
    ]

    texts = [
        str(
            proposal.get(
                field,
                "",
            )
        )
        for field in fields
    ]

    texts.extend(
        str(
            row.get(
                "response",
                "",
            )
        )
        for row in proposal.get(
            "compliance_matrix",
            [],
        )
    )

    unsupported_patterns = [
        r"\bthere are no judgments\b",
        r"\bno pending litigation\b",
        r"\bno expected legal actions\b",
        r"\bwe hereby warrant\b",
        r"\bfully certified\b",
        r"\bfully compliant\b",
    ]

    for text in texts:
        normalized = (
            normalize_match_text(
                text
            )
        )

        for pattern in unsupported_patterns:
            if re.search(
                pattern,
                normalized,
            ):
                errors.append(
                    "Proposal contains an unsupported "
                    f"sensitive company assertion: {text}"
                )
                break

    return errors
##
###
def validate_citations(
    proposal: dict[str, Any],
    proposal_chunks: list[dict[str, Any]],
) -> list[str]:
    warnings = []

    valid_chunk_ids = {
        str(chunk["chunk_id"])
        for chunk in proposal_chunks
        if (
            chunk.get("chunk_id")
            and chunk.get("database_target") == "proposal_db"
        )
    }
    ## new 
    # evidence_catalog = build_evidence_catalog(
    #     proposal_chunks=proposal_chunks
    # )

    # if not evidence_catalog:
    #     raise ValueError(
    #         "No controlled proposal evidence could be created."
    #     )

    # valid_evidence_ids = sorted(
    #     evidence_catalog.keys()
    # )

    # evidence_context = build_evidence_context(
    #     evidence_catalog=evidence_catalog
    # )
    ## new update

    citations = proposal.get("citations", [])
    used_evidence_ids = [
        str(
            citation.get("evidence_id", "")
        ).strip()
        for citation in citations
        if citation.get("evidence_id")
    ]

    evidence_counts = Counter(
        used_evidence_ids
    )

    duplicate_evidence_ids = sorted(
        evidence_id
        for evidence_id, count
        in evidence_counts.items()
        if count > 1
    )

    if duplicate_evidence_ids:
        raise ValueError(
            "Duplicate evidence IDs were reused across "
            "multiple citations: "
            + ", ".join(duplicate_evidence_ids)
        )
    for citation in citations:
        chunk_id = str(
            citation.get("chunk_id", "")
        ).strip()

        if not chunk_id:
            warnings.append(
                "Citation has an empty chunk_id."
            )
            continue

        if chunk_id not in valid_chunk_ids:
            warnings.append(
                f"Citation uses unknown chunk_id: {chunk_id}"
            )

    # Check repeated evidence once, after validating all citations.
    used_chunk_ids = [
        str(citation.get("chunk_id", "")).strip()
        for citation in citations
        if citation.get("chunk_id")
    ]

    if (
        len(used_chunk_ids) >= 3
        and len(set(used_chunk_ids)) == 1
    ):
        warnings.append(
            "All citations use the same proposal chunk. "
            "Inspect whether every claim is genuinely supported."
        )

    return warnings


def print_summary(proposal: dict[str, Any], warnings: list[str]):
    print("\nProposal generation complete.")
    print(f"Title: {proposal.get('title')}")
    print(f"Proposal type: {proposal.get('proposal_type')}")
    print(f"Compliance rows: {len(proposal.get('compliance_matrix', []))}")
    print(f"Citations: {len(proposal.get('citations', []))}")
    print(f"Assumptions: {len(proposal.get('assumptions', []))}")

    print("\nExecutive summary preview:")
    executive_summary = proposal.get("executive_summary", "")
    print(executive_summary[:1000])

    if warnings:
        print("\nCitation warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nCitation check: passed")

def validate_historical_client_leakage(
    proposal: dict[str, Any],
    current_client_name: str,
    historical_client_names: list[str],
) -> list[str]:
    errors = []

    current_client = normalize_match_text(
        current_client_name
    )

    forbidden_terms = {
        normalize_match_text(name)
        for name in historical_client_names
        if (
            normalize_match_text(name)
            and normalize_match_text(name)
            != current_client
        )
    }

    current_client_fields = [
        "executive_summary",
        "understanding_of_requirements",
        "proposed_solution",
        "technical_approach",
        "business_value",
        "implementation_plan",
        "timeline",
        "risk_management",
    ]

    for field in current_client_fields:
        text = normalize_match_text(
            proposal.get(field, "")
        )

        for term in forbidden_terms:
            if term in text:
                errors.append(
                    f"Historical client leakage in "
                    f"{field}: {term}"
                )

    for index, row in enumerate(
        proposal.get("compliance_matrix", []),
        start=1,
    ):
        response = normalize_match_text(
            row.get("response", "")
        )

        for term in forbidden_terms:
            if term in response:
                errors.append(
                    f"Historical client leakage in "
                    f"compliance row {index}: {term}"
                )

    return errors

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--requirements",
        default=str(EXTRACTED_REQUIREMENTS_PATH),
        help="Path to extracted requirements JSON.",
    )

    parser.add_argument(
        "--proposal-context",
        default=str(RETRIEVED_PROPOSAL_CONTEXT_PATH),
        help="Path to retrieved proposal context JSON.",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to save generated proposal JSON.",
    )

    parser.add_argument(
        "--proposal-type",
        choices=["business", "technical", "both"],
        default=None,
        help="Proposal type. Defaults to proposal_type from retrieved proposal context.",
    )

    parser.add_argument(
        "--detail-level",
        choices=["concise", "standard", "detailed"],
        default="standard",
        help="How detailed the generated proposal should be.",
    )

    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name.",
    )

    parser.add_argument(
        "--max-requirements",
        type=int,
        default=DEFAULT_REQUIREMENTS_MAX_ITEMS,
        help="Maximum requirements to send to the model.",
    )

    # parser.add_argument(
    #     # "--max-context-chars",
    #     type=int,
    #     # default=DEFAULT_CONTEXT_MAX_CHARS,
    #     help="Maximum proposal knowledge characters sent to the model.",
    # )

    args = parser.parse_args()

    extracted = load_json(
        Path(args.requirements)
    )

    proposal_context = load_json(
        Path(args.proposal_context)
    )

    requirements = (
        proposal_context.get("selected_requirements")
        or extracted.get("requirements", [])
    )
    valid_requirement_ids = [
        str(
            requirement.get(
                "requirement_id",
                "",
            )
        ).strip()
        for requirement in requirements
        if requirement.get(
            "requirement_id"
        )
    ]

    if not valid_requirement_ids:
        raise ValueError(
            "No valid requirement IDs were found."
        )

    proposal_chunks = proposal_context.get(
        "chunks",
        [],
    )

    current_opportunity_context = (
        build_current_opportunity_context(
            extracted=extracted,
            requirements=requirements,
        )
    )

    if not requirements:
        raise ValueError("No selected requirements found.")

    if not proposal_chunks:
        raise ValueError("No retrieved proposal chunks found.")

    # Only proposal_db chunks are allowed to support company claims.
    valid_chunk_ids = sorted({
        str(chunk["chunk_id"])
        for chunk in proposal_chunks
        if (
            chunk.get("chunk_id")
            and chunk.get("database_target") == "proposal_db"
        )
    })

    if not valid_chunk_ids:
        raise ValueError(
            "No valid proposal_db chunk IDs were found in "
            "retrieved_proposal_context.json."
        )

    candidate_catalog = build_evidence_catalog(
        proposal_chunks=proposal_chunks
    )

    if not candidate_catalog:
        raise ValueError(
            "No proposal evidence candidates "
            "could be created."
        )

    evidence_catalog = filter_approved_evidence(
        candidate_catalog=candidate_catalog,
        approved_path=APPROVED_EVIDENCE_PATH,
    )

    valid_evidence_ids = sorted(
        evidence_catalog.keys()
    )

    evidence_context = build_evidence_context(
        evidence_catalog=evidence_catalog
    )

    proposal_type = (
        args.proposal_type
        or proposal_context.get("proposal_type", "both")
    )
#####
    requirements_context = build_requirements_context(
        requirements=requirements,
        # max_items=args.max_requirements,
        max_items=len(requirements),
    )

    # proposal_knowledge_context = build_proposal_knowledge_context(
    #     chunks=proposal_chunks,
    #     max_chars=args.max_context_chars,
    # )

    print(f"Opportunity ID: {extracted.get('opportunity_id')}")
    print(f"Proposal type: {proposal_type}")
    print(f"Detail level: {args.detail_level}")
    print(f"Requirements sent: {min(len(requirements), args.max_requirements)}")
    print(f"Retrieved proposal chunks available: {len(proposal_chunks)}")
    print(f"Valid proposal citation chunk IDs: {len(valid_chunk_ids)}")
    # print(f"Proposal context characters sent: {len(proposal_knowledge_context)}")
    print(f"Model: {args.model}")

    print(
    f"Controlled evidence options: "
    f"{len(valid_evidence_ids)}"
    )
    print(f"Evidence context characters: "f"{len(evidence_context)}"
    )
    #updated 
    proposal = generate_proposal_with_openai(
        proposal_type=proposal_type,
        current_opportunity_context=current_opportunity_context,
        requirements_context=requirements_context,
        # proposal_knowledge_context=proposal_knowledge_context,
        evidence_context=evidence_context,
        detail_level=args.detail_level,
        model=args.model,
        valid_chunk_ids=valid_chunk_ids,
        valid_evidence_ids=valid_evidence_ids,
        valid_requirement_ids=(
            valid_requirement_ids
        ),        
    )
    ## add to save validation failuer
    ## validation failure
    save_json(proposal,RAW_OUTPUT_PATH,)
    print(f"Saved raw model output to: {RAW_OUTPUT_PATH}")

    ##
    proposal = enrich_and_validate_citations(
        proposal=proposal,
        evidence_catalog=evidence_catalog,
    )

    warnings = validate_citations(
        proposal=proposal,
        proposal_chunks=proposal_chunks,
    )

    citation_type_errors = validate_citation_claim_types(
        proposal
    )

    if citation_type_errors:
        raise ValueError(
            "Invalid citation claim types:\n- "
            + "\n- ".join(citation_type_errors)
        )
    sensitive_claim_errors = (
        validate_sensitive_company_claims(
            proposal
        )
    )

    if sensitive_claim_errors:
        raise ValueError(
            "Unsupported sensitive company claims:\n- "
            + "\n- ".join(
                sensitive_claim_errors
            )
        )
    claim_errors = validate_overbroad_company_claims(
        proposal
    )

    if claim_errors:
        warnings.extend(
            f"Overbroad company claim: {error}"
            for error in claim_errors
        )
    current_client_name = str(
        extracted.get("client_name")
        or extracted.get("organization_name")
        or extracted.get("issuer_name")
        or extracted.get("agency_name")
        or ""
    ).strip()

    historical_client_names = proposal_context.get(
        "historical_client_names",
        [],
    )

    if not isinstance(historical_client_names, list):
        historical_client_names = []
    leakage_errors = validate_historical_client_leakage(
        proposal=proposal,
        current_client_name=current_client_name,
        historical_client_names=historical_client_names,
    )

    if leakage_errors:
        raise ValueError(
            "Historical client leakage detected:\n- "
            + "\n- ".join(leakage_errors)
        )

    save_json(
        proposal,
        Path(args.output),
    )

    print(f"\nSaved generated proposal to: {args.output}")
    print_summary(proposal, warnings)


if __name__ == "__main__":
    main()