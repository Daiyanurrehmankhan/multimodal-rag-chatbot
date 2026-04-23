"""Unified structured logging and global error handling utilities."""

import uuid
from typing import Any

from flask import g, jsonify, request
from werkzeug.exceptions import HTTPException


def _pick_first(*values) -> str | None:
    """Return first non-empty value converted to string."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _safe_json_payload() -> dict[str, Any]:
    """Return JSON payload only when request content is JSON."""
    if not request.is_json:
        return {}
    return request.get_json(silent=True) or {}


def _resolve_request_identity(payload: dict[str, Any]) -> dict[str, str | None]:
    """Resolve correlation identifiers from headers, query, and payload."""
    request_id = _pick_first(
        request.headers.get("X-Request-ID"),
        request.headers.get("X-Correlation-ID"),
        str(uuid.uuid4()),
    )
    session_id = _pick_first(
        request.headers.get("X-Session-ID"),
        request.args.get("session_id"),
        payload.get("session_id"),
    )
    user_id = _pick_first(
        request.headers.get("X-User-ID"),
        request.args.get("user_id"),
        payload.get("user_id"),
    )
    owner_key = _pick_first(
        request.args.get("owner_key"),
        request.args.get("browser_id"),
        payload.get("owner_key"),
        payload.get("browser_id"),
    )
    return {
        "request_id": request_id,
        "session_id": session_id,
        "user_id": user_id,
        "owner_key": owner_key,
    }


def _status_to_error_code(status_code: int) -> str:
    """Map HTTP status to stable machine-readable error code."""
    mapping = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "too_many_requests",
        500: "internal_error",
        502: "bad_gateway",
        503: "service_unavailable",
    }
    return mapping.get(status_code, f"http_{status_code}")


def register_observability(app):
    """Register request/response logging and global error handlers."""

    @app.before_request
    def attach_request_context():
        payload = _safe_json_payload()
        identity = _resolve_request_identity(payload)

        g.request_id = identity["request_id"]
        g.session_id = identity["session_id"]
        g.user_id = identity["user_id"]
        g.owner_key = identity["owner_key"]

        args = dict(request.args) if request.args else {}
        app.logger.info(
            "REQUEST method=%s path=%s request_id=%s session_id=%s user_id=%s owner_key=%s args=%s json=%s",
            request.method,
            request.path,
            g.request_id,
            g.session_id,
            g.user_id,
            g.owner_key,
            args,
            payload if payload else None,
        )

    @app.after_request
    def log_response(response):
        response.headers["X-Request-ID"] = str(getattr(g, "request_id", ""))
        app.logger.info(
            "RESPONSE method=%s path=%s status=%s request_id=%s session_id=%s user_id=%s",
            request.method,
            request.path,
            response.status,
            getattr(g, "request_id", None),
            getattr(g, "session_id", None),
            getattr(g, "user_id", None),
        )
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException):
        status_code = int(exc.code or 500)
        payload = {
            "error": {
                "code": _status_to_error_code(status_code),
                "message": exc.description or "Request failed",
                "details": {"name": exc.name},
            },
            "request_id": getattr(g, "request_id", None),
            "session_id": getattr(g, "session_id", None),
            "user_id": getattr(g, "user_id", None),
        }
        return jsonify(payload), status_code

    @app.errorhandler(Exception)
    def handle_unexpected_exception(exc: Exception):
        app.logger.exception(
            "UNHANDLED_EXCEPTION request_id=%s session_id=%s user_id=%s path=%s",
            getattr(g, "request_id", None),
            getattr(g, "session_id", None),
            getattr(g, "user_id", None),
            request.path,
        )
        payload = {
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "details": {"type": exc.__class__.__name__},
            },
            "request_id": getattr(g, "request_id", None),
            "session_id": getattr(g, "session_id", None),
            "user_id": getattr(g, "user_id", None),
        }
        return jsonify(payload), 500
