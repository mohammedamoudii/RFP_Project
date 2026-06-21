"""Helpers for validating and storing RFP files uploaded through Streamlit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


DEFAULT_UPLOAD_ROOT = Path(
    "data/raw/rfp_uploads"
)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".md",
    ".txt",
}


class UploadedFileLike(Protocol):
    """
    Minimal interface required from a Streamlit
    UploadedFile object.
    """

    name: str

    def getbuffer(self) -> Any:
        """Return the uploaded file bytes as a buffer-like object."""
        ...


@dataclass(frozen=True)
class SavedUpload:
    """Metadata returned after an uploaded file is written to disk."""

    original_name: str
    saved_name: str
    saved_path: Path
    size_bytes: int


def normalize_opportunity_id(
    opportunity_id: str,
) -> str:
    """
    Convert a user-entered opportunity ID into a safe
    folder name.

    Example:
        "Ministry Cloud RFP 2026"
        -> "ministry_cloud_rfp_2026"
    """
    normalized = safe_text(
        opportunity_id
    ).lower()

    normalized = re.sub(
        r"[^a-z0-9_-]+",
        "_",
        normalized,
    )

    normalized = re.sub(
        r"_+",
        "_",
        normalized,
    ).strip("_-")

    if not normalized:
        raise ValueError(
            "Opportunity ID cannot be empty."
        )

    return normalized[:100]


def safe_text(value: Any) -> str:
    """Normalize optional input values into stripped strings."""

    if value is None:
        return ""

    return str(value).strip()


def sanitize_filename(
    filename: str,
) -> str:
    """
    Remove path traversal and unsafe filename characters.
    """
    original = Path(filename).name
    extension = Path(original).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension or 'none'}"
        )

    stem = Path(original).stem

    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stem,
    )

    safe_stem = re.sub(
        r"_+",
        "_",
        safe_stem,
    ).strip("._-")

    if not safe_stem:
        safe_stem = "document"

    return f"{safe_stem[:120]}{extension}"


def save_uploaded_files(
    uploaded_files: Iterable[UploadedFileLike],
    opportunity_id: str,
    upload_root: Path = DEFAULT_UPLOAD_ROOT,
    replace_existing: bool = False,
) -> list[SavedUpload]:
    """
    Save uploaded Streamlit files under:

        data/raw/rfp_uploads/<opportunity_id>/

    The function does not parse or embed documents.
    """
    files = list(uploaded_files)

    if not files:
        raise ValueError(
            "At least one file must be uploaded."
        )

    normalized_id = normalize_opportunity_id(
        opportunity_id
    )

    opportunity_dir = (
        Path(upload_root)
        / normalized_id
    )

    opportunity_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_uploads: list[SavedUpload] = []
    names_in_request: set[str] = set()

    for uploaded_file in files:
        original_name = safe_text(
            uploaded_file.name
        )

        saved_name = sanitize_filename(
            original_name
        )

        lowercase_name = saved_name.lower()

        if lowercase_name in names_in_request:
            raise ValueError(
                "Duplicate filename in this upload: "
                f"{saved_name}"
            )

        names_in_request.add(
            lowercase_name
        )

        file_bytes = bytes(
            uploaded_file.getbuffer()
        )

        if not file_bytes:
            raise ValueError(
                f"Uploaded file is empty: {original_name}"
            )

        target_path = (
            opportunity_dir
            / saved_name
        )

        if (
            target_path.exists()
            and not replace_existing
        ):
            raise FileExistsError(
                f"File already exists: {target_path}. "
                "Enable replacement or use another "
                "opportunity ID."
            )

        # Write to a temporary file first so incomplete uploads do not
        # replace a previously valid source document.
        temporary_path = target_path.with_suffix(
            target_path.suffix + ".part"
        )

        try:
            temporary_path.write_bytes(
                file_bytes
            )

            temporary_path.replace(
                target_path
            )

        finally:
            if temporary_path.exists():
                temporary_path.unlink(
                    missing_ok=True
                )

        saved_uploads.append(
            SavedUpload(
                original_name=original_name,
                saved_name=saved_name,
                saved_path=target_path.resolve(),
                size_bytes=len(file_bytes),
            )
        )

    return saved_uploads

