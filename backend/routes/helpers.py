"""Shared route helper functions."""

from backend.config import ALLOWED_COURSES, STAFF_ROLES


def normalize_selected_course(selected_course, user_role=None):
    """Normalize user-provided course names to a supported canonical label."""
    if isinstance(selected_course, (list, tuple, set, dict)):
        return False

    selected_value = str(selected_course).strip() if selected_course is not None else ""
    if not selected_value:
        return None

    normalized_role = str(user_role or "").strip().lower()
    if normalized_role in STAFF_ROLES:
        return selected_value

    return ALLOWED_COURSES.get(selected_value.lower())


def normalize_role(user_role) -> str:
    """Return a lowercase role value for consistent role checks."""
    return str(user_role or "").strip().lower()
