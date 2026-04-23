"""Document and PDF endpoints."""

from flask import Blueprint, current_app, request, send_file
from backend.services.document_service import (
    delete_document_by_selector,
    list_documents,
    set_document_status,
    set_document_status_simple,
    upload_document,
)
from backend.services.pdf_service import build_pdf_response

document_bp = Blueprint("documents", __name__)


@document_bp.route("/upload_document", methods=["POST"])
def upload_document_route():
    """Upload one document and index it under the selected course."""
    if "file" not in request.files:
        return {"error": "No file part"}, 400
    file = request.files["file"]
    if file.filename == "":
        return {"error": "No file selected"}, 400
    payload, status = upload_document(file=file, form=request.form)
    return payload, status


@document_bp.route("/list_documents", methods=["GET"])
def list_documents_route():
    """Return document metadata rows from documents table."""
    return list_documents()


@document_bp.route("/set_document_status", methods=["POST"])
def set_document_status_route():
    """Update indexed document status by file name or source."""
    data = request.get_json(silent=True) or {}
    return set_document_status(data)


@document_bp.route("/approve_document", methods=["POST"])
def approve_document_route():
    """Approve all chunks for the given document selector."""
    data = request.get_json(silent=True) or {}
    return set_document_status_simple("approved", data)


@document_bp.route("/reject_document", methods=["POST"])
def reject_document_route():
    """Reject all chunks for the given document selector."""
    data = request.get_json(silent=True) or {}
    return set_document_status_simple("rejected", data)


@document_bp.route("/delete_document", methods=["POST"])
def delete_document_route():
    """Delete document chunks and metadata by document_id, file_name, or source."""
    data = request.get_json(silent=True) or {}
    return delete_document_by_selector(data)


@document_bp.route("/download_pdf", methods=["GET"])
def download_pdf():
    """Download a session response as PDF."""
    def pick_value(payload: dict, *keys):
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return value
        return None

    session_id = request.args.get("session_id")
    turn_index = pick_value(request.args, "turn_index")
    message_index = pick_value(request.args, "message_index")
    response_text = pick_value(request.args, "response_text")

    # Canonical selector is turn_index; other selectors are temporary compatibility paths.
    if message_index not in (None, "") and turn_index in (None, ""):
        current_app.logger.warning("COMPAT_PARAM_USED route=/download_pdf param=message_index")
    if response_text not in (None, "") and turn_index in (None, ""):
        current_app.logger.warning("COMPAT_PARAM_USED route=/download_pdf param=response_text")

    buffer, error = build_pdf_response(
        session_id,
        turn_index=turn_index,
        message_index=message_index,
        response_text=response_text,
    )
    if error:
        payload, status = error
        return payload, status

    return send_file(
        buffer,
        as_attachment=True,
        download_name="response.pdf",
        mimetype="application/pdf",
    )
