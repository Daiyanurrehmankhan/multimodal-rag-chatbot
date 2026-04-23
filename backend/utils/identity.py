"""Centralized owner identity resolution and diagnostics logging."""

from flask import current_app


def _clean(value) -> str:
    """Normalize any owner candidate into a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _resolve_owner(candidates: list[tuple[str, object]]) -> tuple[str, str]:
    """Resolve owner key from ordered candidates and report source field."""
    for source, raw in candidates:
        value = _clean(raw)
        if value:
            return value, source
    return "anonymous", "fallback"


def resolve_owner_from_payload(payload: dict | None) -> tuple[str, str]:
    """Resolve owner identity from request JSON payload."""
    data = payload or {}
    return _resolve_owner(
        [
            ("browser_id", data.get("browser_id")),
            ("owner_key", data.get("owner_key")),
            ("user_id", data.get("user_id")),
            ("email", data.get("email")),
            ("session_email", data.get("session_email")),
        ]
    )


def resolve_owner_from_query(query_args) -> tuple[str, str]:
    """Resolve owner identity from query string arguments."""
    return _resolve_owner(
        [
            ("owner_key", query_args.get("owner_key")),
            ("browser_id", query_args.get("browser_id")),
            ("user_id", query_args.get("user_id")),
            ("email", query_args.get("email")),
        ]
    )


def log_owner_resolution(route_name: str, owner_key: str, owner_source: str, user_id=None, user_role=None):
    """Emit standardized owner diagnostics for auditability and debugging."""
    logger = current_app.logger
    if owner_key.lower() == "anonymous":
        if owner_source == "fallback":
            reason = "no owner candidates provided in request"
        else:
            reason = f"client explicitly sent {owner_source}=anonymous"

        logger.warning(
            "OWNER_KEY_DIAGNOSTIC route=%s owner_key=%s source=%s reason=%s user_id=%s user_role=%s",
            route_name,
            owner_key,
            owner_source,
            reason,
            user_id,
            user_role,
        )
        return

    logger.info(
        "OWNER_KEY_DIAGNOSTIC route=%s owner_key=%s source=%s user_id=%s user_role=%s",
        route_name,
        owner_key,
        owner_source,
        user_id,
        user_role,
    )
