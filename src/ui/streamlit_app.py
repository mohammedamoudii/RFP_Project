"""Streamlit UI for the full RFP processing, review, and proposal workflow."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dotenv import load_dotenv


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=True,
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. "
        f"Expected it in: {PROJECT_ROOT / '.env'}"
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.clean_normalize import (
    clean_opportunity_elements,
)
from src.data.create_chunks import (
    create_chunks_for_opportunity,
)
from src.data.create_manifest import (
    create_manifest_for_opportunity,
)
from src.data.insert_chroma import (
    insert_opportunity_chunks,
)
from src.data.parse_documents import (
    parse_opportunity_documents,
)
from src.pipeline.upload_service import (
    normalize_opportunity_id,
    save_uploaded_files,
)
from src.rag.extract_requirements import (
    extract_requirements_from_context,
)
from src.rag.generate_proposal import (
    build_current_opportunity_context,
    build_evidence_catalog,
    build_evidence_context,
    build_requirements_context,
    enrich_and_validate_citations,
    generate_proposal_with_openai,
    validate_citation_claim_types,
    validate_citations,
    validate_historical_client_leakage,
    validate_overbroad_company_claims,
    validate_sensitive_company_claims,
)
from src.rag.list_rfp_opportunities import (
    list_rfp_opportunities,
)
from src.rag.retrieve_proposals import (
    retrieve_proposal_context,
)
from src.rag.retrieve_rfp_context import (
    retrieve_rfp_context,
    save_results as save_retrieval_results,
)
from src.rendering.render_proposal import (
    render_markdown,
)
from src.rendering.export_pdf import (
    export_markdown_to_pdf,
)

PROCESSED_DIR = Path(
    "data/processed"
)

REQUIREMENTS_DIR = (
    PROCESSED_DIR / "requirements"
)

RFP_RETRIEVAL_DIR = (
    PROCESSED_DIR / "retrieval"
)

PROPOSAL_CONTEXT_DIR = (
    PROCESSED_DIR / "proposal_context"
)

EVIDENCE_CANDIDATES_DIR = (
    PROCESSED_DIR / "evidence_candidates"
)

EVIDENCE_APPROVAL_DIR = (
    PROCESSED_DIR / "evidence_approval"
)

GENERATED_DIR = (
    PROCESSED_DIR / "generated"
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load workflow JSON artifacts produced by the pipeline."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def save_json_atomic(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Write JSON artifacts atomically for Streamlit button actions."""

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


def save_text_atomic(
    content: str,
    path: Path,
) -> None:
    """Write text artifacts atomically for generated Markdown output."""

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


def requirements_path(
    opportunity_id: str,
) -> Path:
    """Return the scoped extracted-requirements artifact path."""

    return (
        REQUIREMENTS_DIR
        / (
            f"{opportunity_id}_"
            "extracted_requirements.json"
        )
    )


def proposal_context_path(
    opportunity_id: str,
    proposal_type: str,
) -> Path:
    """Return the scoped proposal-knowledge retrieval path."""

    return (
        PROPOSAL_CONTEXT_DIR
        / (
            f"{opportunity_id}_"
            f"{proposal_type}_"
            "proposal_context.json"
        )
    )


def evidence_candidates_path(
    opportunity_id: str,
    proposal_type: str,
) -> Path:
    """Return the evidence-candidate artifact path for review."""

    return (
        EVIDENCE_CANDIDATES_DIR
        / (
            f"{opportunity_id}_"
            f"{proposal_type}_"
            "evidence_candidates.json"
        )
    )


def evidence_approval_path(
    opportunity_id: str,
    proposal_type: str,
) -> Path:
    """Return the saved evidence-approval artifact path."""

    return (
        EVIDENCE_APPROVAL_DIR
        / (
            f"{opportunity_id}_"
            f"{proposal_type}_"
            "approved_evidence_ids.json"
        )
    )


def generated_json_path(
    opportunity_id: str,
    proposal_type: str,
) -> Path:
    """Return the validated generated-proposal JSON path."""

    return (
        GENERATED_DIR
        / (
            f"{opportunity_id}_"
            f"{proposal_type}_"
            "proposal.json"
        )
    )


def generated_raw_path(
    opportunity_id: str,
    proposal_type: str,
) -> Path:
    """Return the raw model-output JSON path before validation enrichment."""

    return (
        GENERATED_DIR
        / (
            f"{opportunity_id}_"
            f"{proposal_type}_"
            "proposal_raw.json"
        )
    )


def generated_markdown_path(
    opportunity_id: str,
    proposal_type: str,
) -> Path:
    """Return the generated proposal Markdown path."""

    return (
        GENERATED_DIR
        / (
            f"{opportunity_id}_"
            f"{proposal_type}_"
            "proposal.md"
        )
    )

def generated_pdf_path(
    opportunity_id: str,
    proposal_type: str,
) -> Path:
    """Return the generated proposal PDF path."""

    return (
        GENERATED_DIR
        / (
            f"{opportunity_id}_"
            f"{proposal_type}_"
            "proposal.pdf"
        )
    )

def initialize_state() -> None:
    """Initialize Streamlit session keys used across workflow tabs."""

    defaults = {
        "selected_opportunity_id": "",
        "selected_requirement_ids": [],
        "selected_proposal_type": "both",
        "proposal_context": None,
        "evidence_catalog": {},
        "approved_evidence_ids": [],
        "generated_proposal": None,
        "generated_markdown": "",
        "generation_warnings": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[
                key
            ] = value


def requirement_table(
    requirements: list[
        dict[str, Any]
    ],
    selected_ids: list[str],
) -> pd.DataFrame:
    """Build the editable requirement-selection table."""

    selected_set = set(
        selected_ids
    )

    rows = []

    for requirement in requirements:
        requirement_id = str(
            requirement.get(
                "requirement_id",
                "",
            )
        )

        rows.append(
            {
                "Select": (
                    requirement_id
                    in selected_set
                ),
                "ID": requirement_id,
                "Requirement": (
                    requirement.get(
                        "requirement_text",
                        "",
                    )
                ),
                "Category": (
                    requirement.get(
                        "category",
                        "",
                    )
                ),
                "Priority": (
                    requirement.get(
                        "priority",
                        "",
                    )
                ),
                "Proposal type": ", ".join(
                    requirement.get(
                        "proposal_type",
                        [],
                    )
                ),
                "Confidence": (
                    requirement.get(
                        "confidence",
                        "",
                    )
                ),
                "Source": (
                    requirement.get(
                        "source_file",
                        "",
                    )
                ),
                "Page": (
                    requirement.get(
                        "page_number",
                        "",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def evidence_table(
    catalog: dict[
        str,
        dict[str, Any],
    ],
) -> pd.DataFrame:
    """Build the evidence-review table shown in Streamlit."""

    rows = []

    for evidence_id, item in (
        catalog.items()
    ):
        rows.append(
            {
                "Evidence ID": evidence_id,
                "Quote": item.get(
                    "quote",
                    "",
                ),
                "Project": item.get(
                    "project_name",
                    "",
                ),
                "Source": item.get(
                    "source_file",
                    "",
                ),
                "Page": item.get(
                    "page_number",
                    "",
                ),
                "Chunk ID": item.get(
                    "chunk_id",
                    "",
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_compliance_coverage(
    proposal: dict[str, Any],
    requirements: list[
        dict[str, Any]
    ],
) -> list[str]:
    """Check that every selected requirement appears once in the matrix."""

    expected_ids = {
        str(
            requirement.get(
                "requirement_id",
                "",
            )
        )
        for requirement in requirements
        if requirement.get(
            "requirement_id"
        )
    }

    matrix_ids = {
        str(
            row.get(
                "requirement_id",
                "",
            )
        )
        for row in proposal.get(
            "compliance_matrix",
            [],
        )
        if row.get(
            "requirement_id"
        )
    }

    errors = []

    missing = sorted(
        expected_ids - matrix_ids
    )

    unexpected = sorted(
        matrix_ids - expected_ids
    )

    if missing:
        errors.append(
            "Compliance matrix is missing: "
            + ", ".join(missing)
        )

    if unexpected:
        errors.append(
            "Compliance matrix contains unexpected IDs: "
            + ", ".join(unexpected)
        )

    return errors


def run_upload_pipeline(
    opportunity_id: str,
    uploaded_files,
    replace_existing: bool,
    retrieval_top_k: int,
) -> dict[str, Any]:
    """Run upload, parsing, chunking, indexing, retrieval, and extraction."""

    normalized_id = (
        normalize_opportunity_id(
            opportunity_id
        )
    )

    if not uploaded_files:
        raise ValueError(
            "Upload at least one RFP file."
        )

    saved_uploads = save_uploaded_files(
        uploaded_files=uploaded_files,
        opportunity_id=normalized_id,
        replace_existing=replace_existing,
    )

    upload_directory = (
        saved_uploads[0]
        .saved_path
        .parent
    )

    manifest_rows = (
        create_manifest_for_opportunity(
            input_dir=upload_directory,
            opportunity_id=normalized_id,
            database_target="rfp_db",
            merge_existing=True,
        )
    )

    parse_result = (
        parse_opportunity_documents(
            manifest_rows=manifest_rows,
            opportunity_id=normalized_id,
            database_target="rfp_db",
            replace_existing=True,
        )
    )

    parsed_elements = parse_result.get(
        "elements",
        [],
    )

    clean_result = (
        clean_opportunity_elements(
            parsed_elements=parsed_elements,
            opportunity_id=normalized_id,
            database_target="rfp_db",
            replace_existing=True,
        )
    )

    cleaned_elements = clean_result.get(
        "elements",
        [],
    )

    chunk_result = (
        create_chunks_for_opportunity(
            cleaned_elements=cleaned_elements,
            opportunity_id=normalized_id,
            database_target="rfp_db",
            replace_existing=True,
        )
    )

    chunks = chunk_result.get(
        "chunks",
        [],
    )

    insert_result = (
        insert_opportunity_chunks(
            chunks=chunks,
            opportunity_id=normalized_id,
            database_target="rfp_db",
            replace_existing=True,
            show_progress=False,
        )
    )

    retrieval_result = (
        retrieve_rfp_context(
            opportunity_id=normalized_id,
            top_k=int(
                retrieval_top_k
            ),
        )
    )

    rfp_retrieval_path = (
        RFP_RETRIEVAL_DIR
        / (
            f"{normalized_id}_"
            "retrieved_rfp_context.json"
        )
    )

    save_retrieval_results(
        retrieval_result,
        rfp_retrieval_path,
    )

    extracted_requirements = (
        extract_requirements_from_context(
            retrieved_context=(
                retrieval_result
            ),
            output_path=(
                requirements_path(
                    normalized_id
                )
            ),
        )
    )

    return {
        "opportunity_id": normalized_id,
        "saved_uploads": saved_uploads,
        "manifest_rows": manifest_rows,
        "parse_result": parse_result,
        "clean_result": clean_result,
        "chunk_result": chunk_result,
        "insert_result": insert_result,
        "retrieval_result": retrieval_result,
        "requirements_result": (
            extracted_requirements
        ),
        "upload_directory": str(
            upload_directory
        ),
    }


def generate_grounded_proposal(
    extracted: dict[str, Any],
    proposal_context: dict[
        str,
        Any,
    ],
    approved_evidence_ids: list[str],
    proposal_type: str,
    detail_level: str,
    model: str,
) -> tuple[
    dict[str, Any],
    str,
    list[str],
]:
    """Generate, validate, enrich, and render a proposal from approved evidence."""

    requirements = (
        proposal_context.get(
            "selected_requirements"
        )
        or extracted.get(
            "requirements",
            [],
        )
    )

    proposal_chunks = (
        proposal_context.get(
            "chunks",
            [],
        )
    )

    if not requirements:
        raise ValueError(
            "No selected requirements found."
        )

    if not proposal_chunks:
        raise ValueError(
            "No proposal knowledge chunks found."
        )

    candidate_catalog = (
        build_evidence_catalog(
            proposal_chunks=(
                proposal_chunks
            )
        )
    )

    unknown_ids = sorted(
        set(
            approved_evidence_ids
        )
        - set(
            candidate_catalog.keys()
        )
    )

    if unknown_ids:
        raise ValueError(
            "Approved evidence contains "
            "unknown IDs: "
            + ", ".join(unknown_ids)
        )

    evidence_catalog = {
        evidence_id: (
            candidate_catalog[
                evidence_id
            ]
        )
        for evidence_id
        in approved_evidence_ids
    }

    if not evidence_catalog:
        raise ValueError(
            "Approve at least one evidence item."
        )

    valid_chunk_ids = sorted(
        {
            str(
                chunk.get(
                    "chunk_id",
                    "",
                )
            )
            for chunk in proposal_chunks
            if (
                chunk.get(
                    "chunk_id"
                )
                and chunk.get(
                    "database_target"
                )
                == "proposal_db"
                and chunk.get(
                    "collection_name"
                )
                == "proposal_knowledge"
            )
        }
    )

    if not valid_chunk_ids:
        raise ValueError(
            "No valid proposal knowledge "
            "chunk IDs were found."
        )

    current_context = (
        build_current_opportunity_context(
            extracted=extracted,
            requirements=requirements,
        )
    )

    # requirements_context = (
    #     build_requirements_context(
    #         requirements=requirements,
    #         max_items=30,
    #     )
    # )
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

    requirements_context = (
        build_requirements_context(
            requirements=requirements,
            max_items=len(requirements),
        )
    )

    evidence_context = (
        build_evidence_context(
            evidence_catalog=(
                evidence_catalog
            )
        )
    )

    proposal = (
        generate_proposal_with_openai(
            proposal_type=proposal_type,
            current_opportunity_context=(
                current_context
            ),
            requirements_context=(
                requirements_context
            ),
            evidence_context=(
                evidence_context
            ),
            detail_level=detail_level,
            model=model,
            valid_chunk_ids=(
                valid_chunk_ids
            ),
            valid_evidence_ids=sorted(
                evidence_catalog.keys()
            ),
            valid_requirement_ids=(
                valid_requirement_ids
            ),
        )
    )

    raw_path = generated_raw_path(
        extracted[
            "opportunity_id"
        ],
        proposal_type,
    )

    save_json_atomic(
        proposal,
        raw_path,
    )

    proposal = (
        enrich_and_validate_citations(
            proposal=proposal,
            evidence_catalog=(
                evidence_catalog
            ),
        )
    )

    warnings = validate_citations(
        proposal=proposal,
        proposal_chunks=(
            proposal_chunks
        ),
    )

    citation_type_errors = (
        validate_citation_claim_types(
            proposal
        )
    )

    if citation_type_errors:
        raise ValueError(
            "Invalid citation claim types:\n- "
            + "\n- ".join(
                citation_type_errors
            )
        )

    overbroad_errors = (
        validate_overbroad_company_claims(
            proposal
        )
    )

    warnings.extend(
        overbroad_errors
    )

    coverage_errors = (
        validate_compliance_coverage(
            proposal=proposal,
            requirements=requirements,
        )
    )

    if coverage_errors:
        raise ValueError(
            "Compliance coverage validation "
            "failed:\n- "
            + "\n- ".join(
                coverage_errors
            )
        )

    leakage_errors = (
        validate_historical_client_leakage(
            proposal=proposal,
            current_client_name=str(
                extracted.get(
                    "client_name",
                    "",
                )
                or ""
            ),
            historical_client_names=[],
        )
    )

    if leakage_errors:
        raise ValueError(
            "Historical client leakage "
            "detected:\n- "
            + "\n- ".join(
                leakage_errors
            )
        )

    markdown = render_markdown(
        proposal
    )

    return (
        proposal,
        markdown,
        warnings,
    )


initialize_state()

st.set_page_config(
    page_title="RFP Intelligence System",
    page_icon="📄",
    layout="wide",
)

st.title("RFP Intelligence System")
st.caption(
    "Upload an RFP, review requirements, retrieve "
    "approved proposal knowledge, and generate a "
    "grounded proposal with citations."
)

upload_tab, requirements_tab, evidence_tab, generation_tab = (
    st.tabs(
        [
            "1. Upload & Process",
            "2. Review Requirements",
            "3. Review Evidence",
            "4. Generate & Export",
        ]
    )
)


with upload_tab:
    st.subheader(
        "Upload and process a current RFP"
    )

    opportunity_id = st.text_input(
        "Opportunity ID",
        placeholder=(
            "example: ministry_cloud_rfp_2026"
        ),
        key="upload_opportunity_id",
    )

    uploaded_files = st.file_uploader(
        "Upload RFP files",
        type=[
            "pdf",
            "docx",
            "pptx",
            "md",
            "txt",
        ],
        accept_multiple_files=True,
        key="rfp_upload_files",
    )

    upload_col_1, upload_col_2 = (
        st.columns(2)
    )

    with upload_col_1:
        replace_existing = (
            st.checkbox(
                "Replace existing files",
                value=False,
            )
        )

    with upload_col_2:
        rfp_top_k = st.number_input(
            "RFP retrieval top-k",
            min_value=1,
            max_value=20,
            value=8,
        )

    if st.button(
        "Run RFP Processing Pipeline",
        type="primary",
    ):
        try:
            with st.spinner(
                "Running upload, parsing, "
                "cleaning, chunking, retrieval, "
                "and requirement extraction..."
            ):
                pipeline_result = (
                    run_upload_pipeline(
                        opportunity_id=(
                            opportunity_id
                        ),
                        uploaded_files=(
                            uploaded_files
                        ),
                        replace_existing=(
                            replace_existing
                        ),
                        retrieval_top_k=int(
                            rfp_top_k
                        ),
                    )
                )

            active_id = pipeline_result[
                "opportunity_id"
            ]

            st.session_state[
                "selected_opportunity_id"
            ] = active_id

            extracted = pipeline_result[
                "requirements_result"
            ]

            requirement_ids = [
                requirement.get(
                    "requirement_id"
                )
                for requirement in (
                    extracted.get(
                        "requirements",
                        [],
                    )
                )
            ]

            st.session_state[
                "selected_requirement_ids"
            ] = requirement_ids

            metric_1, metric_2, metric_3, metric_4 = (
                st.columns(4)
            )

            metric_1.metric(
                "Parsed elements",
                len(
                    pipeline_result[
                        "parse_result"
                    ].get(
                        "elements",
                        [],
                    )
                ),
            )

            metric_2.metric(
                "Cleaned elements",
                len(
                    pipeline_result[
                        "clean_result"
                    ].get(
                        "elements",
                        [],
                    )
                ),
            )

            metric_3.metric(
                "Chunks",
                len(
                    pipeline_result[
                        "chunk_result"
                    ].get(
                        "chunks",
                        [],
                    )
                ),
            )

            metric_4.metric(
                "Requirements",
                len(
                    extracted.get(
                        "requirements",
                        [],
                    )
                ),
            )

            st.success(
                "RFP processing completed for "
                f"'{active_id}'. Continue to "
                "Review Requirements."
            )

        except Exception as error:
            st.error(
                "RFP processing failed."
            )
            st.exception(error)


with requirements_tab:
    st.subheader(
        "Select an opportunity and review requirements"
    )

    try:
        opportunities = (
            list_rfp_opportunities()
        )
    except Exception as error:
        opportunities = []
        st.error(
            "Could not list RFP opportunities."
        )
        st.exception(error)

    available = [
        item
        for item in opportunities
        if item.get(
            "requirements_available"
        )
    ]

    if not available:
        st.warning(
            "No opportunity has extracted "
            "requirements yet."
        )
    else:
        opportunity_ids = [
            item[
                "opportunity_id"
            ]
            for item in available
        ]

        current_id = st.session_state.get(
            "selected_opportunity_id",
            "",
        )

        default_index = (
            opportunity_ids.index(
                current_id
            )
            if current_id
            in opportunity_ids
            else 0
        )

        selected_opportunity = st.selectbox(
            "Processed opportunity",
            options=opportunity_ids,
            index=default_index,
        )

        selected_record = next(
            item
            for item in available
            if item[
                "opportunity_id"
            ]
            == selected_opportunity
        )

        extracted = load_json(
            Path(
                selected_record[
                    "requirements_path"
                ]
            )
        )

        requirements = extracted.get(
            "requirements",
            [],
        )

        previous_id = st.session_state.get(
            "selected_opportunity_id"
        )

        if (
            previous_id
            != selected_opportunity
        ):
            st.session_state[
                "selected_opportunity_id"
            ] = selected_opportunity

            st.session_state[
                "selected_requirement_ids"
            ] = [
                requirement.get(
                    "requirement_id"
                )
                for requirement
                in requirements
            ]

            st.session_state[
                "proposal_context"
            ] = None

            st.session_state[
                "evidence_catalog"
            ] = {}

            st.session_state[
                "approved_evidence_ids"
            ] = []

            st.session_state[
                "generated_proposal"
            ] = None

            st.session_state[
                "generated_markdown"
            ] = ""

        proposal_type = st.selectbox(
            "Proposal type",
            options=[
                "business",
                "technical",
                "both",
            ],
            index=[
                "business",
                "technical",
                "both",
            ].index(
                st.session_state.get(
                    "selected_proposal_type",
                    "both",
                )
            ),
        )

        current_selected_ids = (
            st.session_state.get(
                "selected_requirement_ids",
                [],
            )
        )

        edited_requirements = (
            st.data_editor(
                requirement_table(
                    requirements=(
                        requirements
                    ),
                    selected_ids=(
                        current_selected_ids
                    ),
                ),
                hide_index=True,
                use_container_width=True,
                disabled=[
                    "ID",
                    "Requirement",
                    "Category",
                    "Priority",
                    "Proposal type",
                    "Confidence",
                    "Source",
                    "Page",
                ],
                column_config={
                    "Select": (
                        st.column_config
                        .CheckboxColumn(
                            "Select",
                            default=True,
                        )
                    )
                },
                key=(
                    "requirement_editor_"
                    + selected_opportunity
                ),
            )
        )

        selected_ids = (
            edited_requirements.loc[
                edited_requirements[
                    "Select"
                ]
                == True,
                "ID",
            ]
            .astype(str)
            .tolist()
        )

        req_metric_1, req_metric_2, req_metric_3 = (
            st.columns(3)
        )

        req_metric_1.metric(
            "Available requirements",
            len(requirements),
        )

        req_metric_2.metric(
            "Selected requirements",
            len(selected_ids),
        )

        req_metric_3.metric(
            "Proposal type",
            proposal_type.title(),
        )

        if st.button(
            "Save Requirement Selection",
            type="primary",
        ):
            if not selected_ids:
                st.error(
                    "Select at least one requirement."
                )
            else:
                st.session_state[
                    "selected_opportunity_id"
                ] = selected_opportunity

                st.session_state[
                    "selected_requirement_ids"
                ] = selected_ids

                st.session_state[
                    "selected_proposal_type"
                ] = proposal_type

                st.session_state[
                    "proposal_context"
                ] = None

                st.session_state[
                    "evidence_catalog"
                ] = {}

                st.session_state[
                    "approved_evidence_ids"
                ] = []

                st.success(
                    "Requirement selection saved. "
                    "Continue to Review Evidence."
                )


with evidence_tab:
    st.subheader(
        "Retrieve and approve proposal knowledge"
    )

    active_id = st.session_state.get(
        "selected_opportunity_id",
        "",
    )

    selected_ids = st.session_state.get(
        "selected_requirement_ids",
        [],
    )

    proposal_type = st.session_state.get(
        "selected_proposal_type",
        "both",
    )

    if not active_id:
        st.warning(
            "Select an opportunity in "
            "Review Requirements first."
        )
    elif not selected_ids:
        st.warning(
            "Select at least one requirement first."
        )
    else:
        st.write(
            "**Opportunity:**",
            active_id,
        )

        st.write(
            "**Proposal type:**",
            proposal_type,
        )

        proposal_top_k = st.number_input(
            "Proposal retrieval top-k",
            min_value=1,
            max_value=15,
            value=5,
            key="proposal_top_k",
        )

        max_requirement_queries = (
            st.number_input(
                "Maximum requirement queries",
                min_value=1,
                max_value=200,
                value=12,
                key="max_requirement_queries",
            )
        )

        if st.button(
            "Retrieve Proposal Knowledge",
            type="primary",
        ):
            try:
                extracted = load_json(
                    requirements_path(
                        active_id
                    )
                )

                output_path = (
                    proposal_context_path(
                        active_id,
                        proposal_type,
                    )
                )

                with st.spinner(
                    "Searching proposal knowledge..."
                ):
                    context = (
                        retrieve_proposal_context(
                            extracted_requirements=(
                                extracted
                            ),
                            proposal_type=(
                                proposal_type
                            ),
                            selected_requirement_ids=(
                                selected_ids
                            ),
                            top_k=int(
                                proposal_top_k
                            ),
                            max_requirement_queries=int(
                                max_requirement_queries
                            ),
                            output_path=(
                                output_path
                            ),
                        )
                    )

                catalog = (
                    build_evidence_catalog(
                        proposal_chunks=(
                            context.get(
                                "chunks",
                                [],
                            )
                        )
                    )
                )

                save_json_atomic(
                    {
                        "opportunity_id": (
                            active_id
                        ),
                        "proposal_type": (
                            proposal_type
                        ),
                        "evidence": list(
                            catalog.values()
                        ),
                    },
                    evidence_candidates_path(
                        active_id,
                        proposal_type,
                    ),
                )

                st.session_state[
                    "proposal_context"
                ] = context

                st.session_state[
                    "evidence_catalog"
                ] = catalog

                st.session_state[
                    "approved_evidence_ids"
                ] = []

                st.success(
                    f"Retrieved "
                    f"{context.get('total_retrieved_chunks_after_dedup', 0)} "
                    "unique proposal chunks and created "
                    f"{len(catalog)} evidence candidates."
                )

            except Exception as error:
                st.error(
                    "Proposal knowledge retrieval failed."
                )
                st.exception(error)

        context = st.session_state.get(
            "proposal_context"
        )

        catalog = st.session_state.get(
            "evidence_catalog",
            {},
        )

        if context:
            retrieval_metric_1, retrieval_metric_2, retrieval_metric_3 = (
                st.columns(3)
            )

            retrieval_metric_1.metric(
                "Selected requirements",
                context.get(
                    "selected_requirements_count",
                    0,
                ),
            )

            retrieval_metric_2.metric(
                "Queries",
                context.get(
                    "queries_count",
                    0,
                ),
            )

            retrieval_metric_3.metric(
                "Unique chunks",
                context.get(
                    "total_retrieved_chunks_after_dedup",
                    0,
                ),
            )

        if catalog:
            st.warning(
                "Approve only documented company "
                "capabilities, past experience, named "
                "staff experience, methodologies, or "
                "service commitments. Reject RFP "
                "instructions and evaluation criteria."
            )

            candidate_frame = (
                evidence_table(
                    catalog
                )
            )

            st.dataframe(
                candidate_frame,
                hide_index=True,
                use_container_width=True,
            )

            evidence_ids = list(
                catalog.keys()
            )

            approved_ids = st.multiselect(
                "Approved evidence IDs",
                options=evidence_ids,
                default=[
                    evidence_id
                    for evidence_id in (
                        st.session_state.get(
                            "approved_evidence_ids",
                            [],
                        )
                    )
                    if evidence_id
                    in evidence_ids
                ],
                format_func=lambda evidence_id: (
                    f"{evidence_id} — "
                    f"{catalog[evidence_id]['quote'][:120]}"
                ),
            )

            selected_evidence_rows = [
                catalog[
                    evidence_id
                ]
                for evidence_id in (
                    approved_ids
                )
            ]

            if selected_evidence_rows:
                st.write(
                    "**Selected evidence preview**"
                )

                st.dataframe(
                    pd.DataFrame(
                        selected_evidence_rows
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            if st.button(
                "Save Evidence Approval",
                type="primary",
            ):
                if not approved_ids:
                    st.error(
                        "Approve at least one evidence item."
                    )
                else:
                    approval_payload = {
                        "opportunity_id": (
                            active_id
                        ),
                        "proposal_type": (
                            proposal_type
                        ),
                        "approved_evidence_ids": (
                            approved_ids
                        ),
                    }

                    save_json_atomic(
                        approval_payload,
                        evidence_approval_path(
                            active_id,
                            proposal_type,
                        ),
                    )

                    st.session_state[
                        "approved_evidence_ids"
                    ] = approved_ids

                    st.success(
                        "Evidence approval saved. "
                        "Continue to Generate & Export."
                    )


with generation_tab:
    st.subheader(
        "Generate, preview, and export proposal"
    )

    active_id = st.session_state.get(
        "selected_opportunity_id",
        "",
    )

    proposal_type = st.session_state.get(
        "selected_proposal_type",
        "both",
    )

    approved_ids = st.session_state.get(
        "approved_evidence_ids",
        [],
    )

    context = st.session_state.get(
        "proposal_context"
    )

    if not active_id:
        st.warning(
            "Select an opportunity first."
        )
    elif not context:
        st.warning(
            "Retrieve proposal knowledge first."
        )
    elif not approved_ids:
        st.warning(
            "Approve proposal evidence first."
        )
    else:
        detail_level = st.selectbox(
            "Detail level",
            options=[
                "concise",
                "standard",
                "detailed",
            ],
            index=1,
        )

        model = st.text_input(
            "OpenAI model",
            value=os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini",
            ),
        )

        if st.button(
            "Generate Grounded Proposal",
            type="primary",
        ):
            try:
                extracted = load_json(
                    requirements_path(
                        active_id
                    )
                )

                with st.spinner(
                    "Generating and validating "
                    "the grounded proposal..."
                ):
                    proposal, markdown, warnings = (
                        generate_grounded_proposal(
                            extracted=extracted,
                            proposal_context=context,
                            approved_evidence_ids=(
                                approved_ids
                            ),
                            proposal_type=(
                                proposal_type
                            ),
                            detail_level=(
                                detail_level
                            ),
                            model=model,
                        )
                    )

                json_path = (
                    generated_json_path(
                        active_id,
                        proposal_type,
                    )
                )

                markdown_path = (
                    generated_markdown_path(
                        active_id,
                        proposal_type,
                    )
                )

                save_json_atomic(
                    proposal,
                    json_path,
                )

                save_text_atomic(
                    markdown,
                    markdown_path,
                )

                st.session_state[
                    "generated_proposal"
                ] = proposal

                st.session_state[
                    "generated_markdown"
                ] = markdown

                st.session_state[
                    "generation_warnings"
                ] = warnings

                st.success(
                    "Proposal generated, validated, "
                    "and rendered to Markdown."
                )

            except Exception as error:
                st.error(
                    "Proposal generation failed."
                )
                st.exception(error)

        proposal = st.session_state.get(
            "generated_proposal"
        )

        markdown = st.session_state.get(
            "generated_markdown",
            "",
        )

        warnings = st.session_state.get(
            "generation_warnings",
            [],
        )

        if proposal and markdown:
            metric_1, metric_2, metric_3 = (
                st.columns(3)
            )

            metric_1.metric(
                "Compliance rows",
                len(
                    proposal.get(
                        "compliance_matrix",
                        [],
                    )
                ),
            )

            metric_2.metric(
                "Citations",
                len(
                    proposal.get(
                        "citations",
                        [],
                    )
                ),
            )

            metric_3.metric(
                "Assumptions",
                len(
                    proposal.get(
                        "assumptions",
                        [],
                    )
                ),
            )

            if warnings:
                st.warning(
                    "Generation completed with "
                    f"{len(warnings)} warning(s)."
                )

                for warning in warnings:
                    st.write(
                        "-",
                        warning,
                    )

            st.markdown(
                markdown
            )

            download_col_1, download_col_2, download_col_3 = (
                st.columns(3)
            )

            with download_col_1:
                st.download_button(
                    "Download Proposal JSON",
                    data=json.dumps(
                        proposal,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    file_name=(
                        f"{active_id}_"
                        f"{proposal_type}_"
                        "proposal.json"
                    ),
                    mime=(
                        "application/json"
                    ),
                )

            with download_col_2:
                st.download_button(
                    "Download Proposal Markdown",
                    data=markdown,
                    file_name=(
                        f"{active_id}_"
                        f"{proposal_type}_"
                        "proposal.md"
                    ),
                    mime="text/markdown",
                )

            with download_col_3:
                try:
                    pdf_bytes = export_markdown_to_pdf(
                        markdown_text=markdown,
                        output_path=generated_pdf_path(
                            active_id,
                            proposal_type,
                        ),
                    )

                    st.download_button(
                        "Download Proposal PDF",
                        data=pdf_bytes,
                        file_name=(
                            f"{active_id}_"
                            f"{proposal_type}_"
                            "proposal.pdf"
                        ),
                        mime="application/pdf",
                    )

                except Exception as error:
                    st.error(
                        "PDF export is not available on this machine."
                    )
                    st.caption(
                        "Markdown download is still available. "
                        "Install markdown and weasyprint, then restart Streamlit."
                    )
                    st.exception(error)