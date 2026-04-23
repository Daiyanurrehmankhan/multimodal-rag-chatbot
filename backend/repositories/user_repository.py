"""Repository functions for user and enrollment persistence access."""

from rag_working import (
    add_user_to_course,
    create_user,
    get_all_users,
    get_courses,
    get_user_by_id,
    get_user_courses,
    verify_user,
)


def verify_credentials(email: str, password: str) -> dict | None:
    """Verify credentials and return user when valid."""
    return verify_user(email, password)


def create_account(name: str, email: str, password: str, role: str) -> dict | None:
    """Create user account and return new user when successful."""
    return create_user(name, email, password, role)


def fetch_user(user_id: int) -> dict | None:
    """Fetch one user by id."""
    return get_user_by_id(user_id)


def fetch_users(role: str | None = None) -> list[dict]:
    """Fetch active users optionally filtered by role."""
    return get_all_users(role)


def fetch_user_courses(user_id: int) -> list[dict]:
    """Fetch enrolled courses for one user."""
    return get_user_courses(user_id)


def enroll_user(user_id: int, course_id: int) -> bool:
    """Enroll user into one course."""
    return add_user_to_course(user_id, course_id)


def fetch_courses() -> list[dict]:
    """Fetch all courses used by the frontend course selector."""
    return get_courses()
