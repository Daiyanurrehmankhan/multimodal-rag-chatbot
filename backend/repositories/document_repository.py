"""Repository functions for document persistence access."""

from rag_working import (
    delete_document,
    get_courses,
    index_single_document,
    list_documents_metadata,
    update_document_status,
)


def list_courses() -> list[dict]:
    """Fetch available courses."""
    return get_courses()


def index_document(
    save_path: str,
    selected_course,
    initial_status: str,
    original_filename: str,
    short_description: str,
    generated_course_id: int | None,
) -> tuple[bool, str, int]:
    """Persist uploaded document chunks and metadata."""
    return index_single_document(
        save_path,
        selected_course,
        initial_status=initial_status,
        original_filename=original_filename,
        short_description=short_description,
        generated_course_id=generated_course_id,
    )


def list_document_metadata() -> list[dict]:
    """Fetch document metadata rows."""
    return list_documents_metadata()


def update_status(status: str, file_name: str | None = None, source: str | None = None) -> int:
    """Update document status for matching selectors."""
    return update_document_status(status=status, file_name=file_name, source=source)


def delete_document_rows(document_id=None, file_name=None, source=None) -> int:
    """Delete document chunks and metadata rows by selectors."""
    return delete_document(document_id=document_id, file_name=file_name, source=source)
