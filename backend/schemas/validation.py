"""Shared validation helpers for route/service boundaries."""


def parse_optional_int(value, field_name: str):
    """Parse optional integer values from incoming payloads."""
    if value in (None, ""):
        return None, None
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, {"error": f"{field_name} must be an integer"}


def require_dict(payload, field_name: str = "payload"):
    """Validate that payload is a dictionary-like object."""
    if isinstance(payload, dict):
        return payload, None
    return None, {"error": f"{field_name} must be a JSON object"}
