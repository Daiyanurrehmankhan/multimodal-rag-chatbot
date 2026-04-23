"""Business logic for document operations and PDF export."""

import os
import uuid

from backend.config import UPLOAD_FOLDER
from backend.repositories.document_repository import (
    delete_document_rows,
    index_document,
    list_courses,
    list_document_metadata,
    update_status,
)
from backend.schemas import build_delete_response, build_update_response, parse_optional_int


def upload_document(file, form) -> tuple[dict, int]:
    """Upload one document and index it under the selected course."""
    course_id_raw = form.get("course_id")
    course_name = str(form.get("course_name") or "").strip()
    generated_course_id_raw = form.get("generated_course_id") or form.get("course_generated_id")
    short_description = str(
        form.get("short_description") or form.get("description") or ""
    ).strip()

    generated_course_id, error = parse_optional_int(generated_course_id_raw, "generated_course_id")
    if error:
        return error, 400

    selected_course = None
    if course_name:
        selected_course = course_name
    elif course_id_raw:
        course_id, error = parse_optional_int(course_id_raw, "course_id")
        if error:
            return error, 400
        if course_id is None:
            return {"error": "course_id must be an integer"}, 400

        valid_course_ids = {int(c["id"]) for c in list_courses() if "id" in c}
        if course_id not in valid_course_ids:
            return {"error": "Invalid course_id for upload"}, 400
        selected_course = course_id
    else:
        return {"error": "course_id or course_name is required"}, 400

    uploader_role = str(form.get("uploaded_by_role") or "").strip().lower()
    if uploader_role not in ("admin", "instructor", "student"):
        uploader_role = "instructor"

    safe_name = os.path.basename(file.filename) or "document"
    ext = os.path.splitext(safe_name)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    try:
        file.save(save_path)
    except Exception as exc:
        return {"error": f"Failed to save file: {exc}"}, 500

    initial_status = "approved" if uploader_role in ("admin", "instructor") else "pending"
    success, message, document_id = index_document(
        save_path=save_path,
        selected_course=selected_course,
        initial_status=initial_status,
        original_filename=safe_name,
        short_description=short_description,
        generated_course_id=generated_course_id,
    )
    if not success:
        return {"error": message, "document_id": document_id}, 400
    return {"status": "indexed", "message": message, "document_id": document_id}, 200


def list_documents() -> tuple[dict, int]:
    """Return document metadata rows."""
    try:
        docs = list_document_metadata()
        return {"documents": docs}, 200
    except Exception as exc:
        return {"error": str(exc)}, 500


def set_document_status(data: dict) -> tuple[dict, int]:
    """Update indexed document status by file name or source."""
    status = str(data.get("status") or "").strip().lower()
    file_name = data.get("file_name")
    source = data.get("source")

    if status not in ("pending", "approved", "rejected"):
        return {"error": "status must be one of pending, approved, rejected"}, 400

    if not file_name and not source:
        return {"error": "Provide file_name or source"}, 400

    updated_count = update_status(status=status, file_name=file_name, source=source)
    response = build_update_response(updated_count)
    if response["updated_count"] == 0:
        return response, 404

    return response, 200


def set_document_status_simple(status: str, data: dict) -> tuple[dict, int]:
    """Update document status for approve/reject helper routes."""
    updated_count = update_status(
        status=status,
        file_name=data.get("file_name"),
        source=data.get("source"),
    )
    response = build_update_response(updated_count)
    if response["updated_count"] == 0:
        return response, 404
    return response, 200


def delete_document_by_selector(data: dict) -> tuple[dict, int]:
    """Delete document chunks and metadata by selectors."""
    document_id = data.get("document_id")
    file_name = data.get("file_name")
    source = data.get("source")

    if document_id is None and not file_name and not source:
        return {"error": "Provide document_id or file_name or source"}, 400

    if document_id is not None:
        parsed_document_id, error = parse_optional_int(document_id, "document_id")
        if error:
            return error, 400
        document_id = parsed_document_id

    deleted_count = delete_document_rows(document_id=document_id, file_name=file_name, source=source)
    response = build_delete_response(deleted_count)
    if response["deleted_count"] == 0:
        return response, 404

    return response, 200


