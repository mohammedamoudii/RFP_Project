"""Streamlit smoke-test page for the end-to-end RFP upload pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import streamlit as st

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
from src.rag.retrieve_rfp_context import (
    retrieve_rfp_context,
    save_results as save_retrieval_results,
)


st.set_page_config(
    page_title="RFP Pipeline",
    page_icon="📄",
    layout="wide",
)

st.title("RFP Intelligence System")
st.header(
    "Phase 13B8 — End-to-End RFP Processing"
)

st.info(
    "This page uploads the active RFP, creates its "
    "manifest, parses and cleans the text, creates chunks, "
    "stores them in ChromaDB, retrieves requirement context, "
    "and extracts structured requirements."
)

st.caption(
    "Requirement extraction sends retrieved RFP context "
    "to the OpenAI model configured in your environment."
)

opportunity_id = st.text_input(
    "Opportunity ID",
    placeholder=(
        "example: ministry_cloud_rfp_2026"
    ),
    help=(
        "Use the same opportunity ID for all files "
        "belonging to one RFP."
    ),
)

uploaded_files = st.file_uploader(
    "Upload current RFP files",
    type=[
        "pdf",
        "docx",
        "pptx",
        "md",
        "txt",
    ],
    accept_multiple_files=True,
)

control_col_1, control_col_2 = (
    st.columns(2)
)

with control_col_1:
    replace_existing = st.checkbox(
        "Replace files that already exist",
        value=False,
    )

with control_col_2:
    retrieval_top_k = st.number_input(
        "Top-k chunks per retrieval query",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
    )


if st.button(
    "Run Full RFP Pipeline",
    type="primary",
):
    current_stage = "validating input"

    try:
        normalized_id = (
            normalize_opportunity_id(
                opportunity_id
            )
        )

        if not uploaded_files:
            raise ValueError(
                "Upload at least one RFP file."
            )

        current_stage = "saving uploaded files"

        with st.spinner(
            "Stage 1/8 — Saving uploaded files..."
        ):
            saved_uploads = (
                save_uploaded_files(
                    uploaded_files=(
                        uploaded_files
                    ),
                    opportunity_id=(
                        normalized_id
                    ),
                    replace_existing=(
                        replace_existing
                    ),
                )
            )

            upload_directory = (
                saved_uploads[0]
                .saved_path
                .parent
            )

        current_stage = "creating manifest"

        with st.spinner(
            "Stage 2/8 — Creating manifest..."
        ):
            manifest_rows = (
                create_manifest_for_opportunity(
                    input_dir=(
                        upload_directory
                    ),
                    opportunity_id=(
                        normalized_id
                    ),
                    database_target=(
                        "rfp_db"
                    ),
                    merge_existing=True,
                )
            )

        current_stage = "parsing documents"

        with st.spinner(
            "Stage 3/8 — Parsing documents..."
        ):
            parse_result = (
                parse_opportunity_documents(
                    manifest_rows=(
                        manifest_rows
                    ),
                    opportunity_id=(
                        normalized_id
                    ),
                    database_target=(
                        "rfp_db"
                    ),
                    replace_existing=True,
                )
            )

            active_parsed_elements = (
                parse_result.get(
                    "elements",
                    [],
                )
            )

        current_stage = (
            "cleaning parsed elements"
        )

        with st.spinner(
            "Stage 4/8 — Cleaning text..."
        ):
            clean_result = (
                clean_opportunity_elements(
                    parsed_elements=(
                        active_parsed_elements
                    ),
                    opportunity_id=(
                        normalized_id
                    ),
                    database_target=(
                        "rfp_db"
                    ),
                    replace_existing=True,
                )
            )

            active_cleaned_elements = (
                clean_result.get(
                    "elements",
                    [],
                )
            )

        current_stage = "creating chunks"

        with st.spinner(
            "Stage 5/8 — Creating chunks..."
        ):
            chunk_result = (
                create_chunks_for_opportunity(
                    cleaned_elements=(
                        active_cleaned_elements
                    ),
                    opportunity_id=(
                        normalized_id
                    ),
                    database_target=(
                        "rfp_db"
                    ),
                    replace_existing=True,
                )
            )

            active_chunks = (
                chunk_result.get(
                    "chunks",
                    [],
                )
            )

        current_stage = (
            "embedding and inserting into ChromaDB"
        )

        with st.spinner(
            "Stage 6/8 — Embedding and storing chunks..."
        ):
            insert_result = (
                insert_opportunity_chunks(
                    chunks=active_chunks,
                    opportunity_id=(
                        normalized_id
                    ),
                    database_target=(
                        "rfp_db"
                    ),
                    replace_existing=True,
                    show_progress=False,
                )
            )

        current_stage = (
            "retrieving RFP requirement context"
        )

        with st.spinner(
            "Stage 7/8 — Retrieving requirement context..."
        ):
            retrieval_result = (
                retrieve_rfp_context(
                    opportunity_id=(
                        normalized_id
                    ),
                    top_k=int(
                        retrieval_top_k
                    ),
                )
            )

            retrieval_output_path = (
                Path(
                    "data/processed/retrieval"
                )
                / (
                    f"{normalized_id}_"
                    "retrieved_rfp_context.json"
                )
            )

            save_retrieval_results(
                retrieval_result,
                retrieval_output_path,
            )

        current_stage = (
            "extracting structured requirements"
        )

        with st.spinner(
            "Stage 8/8 — Extracting requirements..."
        ):
            requirements_output_path = (
                Path(
                    "data/processed/requirements"
                )
                / (
                    f"{normalized_id}_"
                    "extracted_requirements.json"
                )
            )

            requirements_result = (
                extract_requirements_from_context(
                    retrieved_context=(
                        retrieval_result
                    ),
                    output_path=(
                        requirements_output_path
                    ),
                )
            )

            active_requirements = (
                requirements_result.get(
                    "requirements",
                    [],
                )
            )

        state_values = {
            "active_opportunity_id": (
                normalized_id
            ),
            "active_upload_directory": (
                str(upload_directory)
            ),
            "active_manifest_rows": (
                manifest_rows
            ),
            "active_parse_result": (
                parse_result
            ),
            "active_parsed_elements": (
                active_parsed_elements
            ),
            "active_clean_result": (
                clean_result
            ),
            "active_cleaned_elements": (
                active_cleaned_elements
            ),
            "active_chunk_result": (
                chunk_result
            ),
            "active_chunks": (
                active_chunks
            ),
            "active_insert_result": (
                insert_result
            ),
            "active_retrieval_result": (
                retrieval_result
            ),
            "active_requirements_result": (
                requirements_result
            ),
        }

        for key, value in (
            state_values.items()
        ):
            st.session_state[
                key
            ] = value

        st.success(
            f"Completed the pipeline for "
            f"'{normalized_id}'. "
            f"Generated {len(active_chunks)} chunks, "
            f"retrieved "
            f"{retrieval_result.get('total_chunks', 0)} "
            f"unique context chunks, and extracted "
            f"{len(active_requirements)} requirements."
        )

        st.subheader("Saved Uploads")

        upload_rows = [
            {
                "Original filename": (
                    item.original_name
                ),
                "Saved filename": (
                    item.saved_name
                ),
                "Size (bytes)": (
                    item.size_bytes
                ),
                "Saved path": str(
                    item.saved_path
                ),
            }
            for item in saved_uploads
        ]

        st.dataframe(
            upload_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Manifest Records")

        st.dataframe(
            manifest_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Parsing Results")

        parse_col_1, parse_col_2, parse_col_3 = (
            st.columns(3)
        )

        parse_col_1.metric(
            "Files selected",
            parse_result.get(
                "files_selected",
                0,
            ),
        )

        parse_col_2.metric(
            "Parsed elements",
            len(
                active_parsed_elements
            ),
        )

        parse_col_3.metric(
            "Parsing errors",
            parse_result.get(
                "error_count",
                0,
            ),
        )

        st.subheader("Cleaning Results")

        clean_col_1, clean_col_2, clean_col_3 = (
            st.columns(3)
        )

        clean_col_1.metric(
            "Input elements",
            clean_result.get(
                "input_element_count",
                0,
            ),
        )

        clean_col_2.metric(
            "Cleaned elements",
            len(
                active_cleaned_elements
            ),
        )

        clean_col_3.metric(
            "Removed elements",
            clean_result.get(
                "removed_element_count",
                0,
            ),
        )

        st.subheader("Chunking Results")

        chunk_col_1, chunk_col_2, chunk_col_3 = (
            st.columns(3)
        )

        chunk_col_1.metric(
            "Input cleaned elements",
            chunk_result.get(
                "input_element_count",
                0,
            ),
        )

        chunk_col_2.metric(
            "Generated chunks",
            len(active_chunks),
        )

        chunk_col_3.metric(
            "Chunk overlap",
            chunk_result.get(
                "chunk_overlap",
                0,
            ),
        )

        if active_chunks:
            chunk_preview_rows = [
                {
                    "Chunk ID": chunk.get(
                        "chunk_id",
                        "",
                    ),
                    "Source": chunk.get(
                        "source_file",
                        "",
                    ),
                    "Pages": chunk.get(
                        "page_numbers",
                        "",
                    ),
                    "Words": chunk.get(
                        "chunk_word_count",
                        0,
                    ),
                    "Preview": str(
                        chunk.get(
                            "content",
                            "",
                        )
                    )[:250],
                }
                for chunk in active_chunks[:10]
            ]

            st.dataframe(
                chunk_preview_rows,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(
            "ChromaDB Insertion"
        )

        insert_col_1, insert_col_2, insert_col_3 = (
            st.columns(3)
        )

        insert_col_1.metric(
            "Inserted opportunity chunks",
            insert_result.get(
                "input_chunk_count",
                0,
            ),
        )

        insert_col_2.metric(
            "Stored scope count",
            insert_result.get(
                "stored_scope_count",
                0,
            ),
        )

        insert_col_3.metric(
            "Total collection count",
            insert_result.get(
                "collection_count",
                0,
            ),
        )

        st.write(
            "**Collection:**",
            insert_result.get(
                "collection_name"
            ),
        )

        st.write(
            "**Embedding model:**",
            insert_result.get(
                "embedding_model"
            ),
        )

        st.subheader(
            "RFP Context Retrieval"
        )

        retrieval_col_1, retrieval_col_2, retrieval_col_3 = (
            st.columns(3)
        )

        retrieval_col_1.metric(
            "Queries",
            len(
                retrieval_result.get(
                    "queries",
                    [],
                )
            ),
        )

        retrieval_col_2.metric(
            "Raw results",
            retrieval_result.get(
                "raw_result_count",
                0,
            ),
        )

        retrieval_col_3.metric(
            "Unique chunks",
            retrieval_result.get(
                "total_chunks",
                0,
            ),
        )

        retrieved_preview_rows = [
            {
                "Rank": chunk.get(
                    "combined_rank",
                    chunk.get("rank"),
                ),
                "Chunk ID": chunk.get(
                    "chunk_id",
                    "",
                ),
                "Distance": chunk.get(
                    "distance",
                ),
                "Pages": chunk.get(
                    "page_numbers",
                    "",
                ),
                "Source": chunk.get(
                    "source_file",
                    "",
                ),
                "Preview": chunk.get(
                    "content_preview",
                    "",
                ),
            }
            for chunk in retrieval_result.get(
                "chunks",
                [],
            )[:10]
        ]

        if retrieved_preview_rows:
            st.dataframe(
                retrieved_preview_rows,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(
            "Extracted Requirements"
        )

        validation = (
            requirements_result.get(
                "validation",
                {},
            )
        )

        req_col_1, req_col_2, req_col_3 = (
            st.columns(3)
        )

        req_col_1.metric(
            "Requirements",
            len(
                active_requirements
            ),
        )

        req_col_2.metric(
            "Valid citations",
            validation.get(
                "valid_chunk_citations",
                0,
            ),
        )

        req_col_3.metric(
            "Citation warnings",
            len(
                validation.get(
                    "page_warnings",
                    [],
                )
            )
            + len(
                validation.get(
                    "evidence_quote_warnings",
                    [],
                )
            ),
        )

        requirement_rows = [
            {
                "ID": requirement.get(
                    "requirement_id"
                ),
                "Requirement": requirement.get(
                    "requirement_text"
                ),
                "Category": requirement.get(
                    "category"
                ),
                "Priority": requirement.get(
                    "priority"
                ),
                "Proposal type": ", ".join(
                    requirement.get(
                        "proposal_type",
                        [],
                    )
                ),
                "Confidence": requirement.get(
                    "confidence"
                ),
                "Source": requirement.get(
                    "source_file"
                ),
                "Page": requirement.get(
                    "page_number"
                ),
                "Chunk ID": requirement.get(
                    "chunk_id"
                ),
            }
            for requirement in (
                active_requirements
            )
        ]

        st.dataframe(
            requirement_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download retrieved context JSON",
            data=json.dumps(
                retrieval_result,
                ensure_ascii=False,
                indent=2,
            ),
            file_name=(
                f"{normalized_id}_"
                "retrieved_rfp_context.json"
            ),
            mime="application/json",
        )

        st.download_button(
            "Download requirements JSON",
            data=json.dumps(
                requirements_result,
                ensure_ascii=False,
                indent=2,
            ),
            file_name=(
                f"{normalized_id}_"
                "extracted_requirements.json"
            ),
            mime="application/json",
        )

        st.subheader(
            "Active Opportunity"
        )

        st.write(
            "**Opportunity ID:**",
            normalized_id,
        )

        st.write(
            "**Upload directory:**",
            str(upload_directory),
        )

        st.write(
            "**Retrieval output:**",
            str(
                retrieval_output_path
            ),
        )

        st.write(
            "**Requirements output:**",
            str(
                requirements_output_path
            ),
        )

    except Exception as error:
        st.error(
            f"The pipeline failed during "
            f"'{current_stage}'."
        )

        st.exception(error)
