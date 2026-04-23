"""Business logic for auth, users, and enrollments."""

from backend.repositories.user_repository import (
    create_account,
    enroll_user,
    fetch_courses,
    fetch_user,
    fetch_user_courses,
    fetch_users,
    verify_credentials,
)
from backend.schemas import parse_optional_int, require_dict


def login(data: dict) -> tuple[dict, int]:
    """Authenticate user with email/password payload."""
    data, error = require_dict(data)
    if error:
        return error, 400

    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        return {"error": "email and password are required"}, 400

    user = verify_credentials(email, password)
    if user:
        return {"status": "success", "user": user}, 200
    return {"error": "Invalid email or password"}, 401


def register(data: dict) -> tuple[dict, int]:
    """Register a new user from payload."""
    data, error = require_dict(data)
    if error:
        return error, 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    role = data.get("role", "student").strip().lower()

    if not all([name, email, password]):
        return {"error": "name, email, and password are required"}, 400

    if role not in ["student", "instructor", "admin"]:
        return {"error": "role must be student, instructor, or admin"}, 400

    user = create_account(name, email, password, role)
    if user:
        return {"status": "success", "user": user}, 201
    return {"error": "Email already exists or registration failed"}, 400


def get_user(user_id: int) -> tuple[dict, int]:
    """Return one user by id."""
    user = fetch_user(user_id)
    if user:
        return {"user": user}, 200
    return {"error": "User not found"}, 404


def list_users(role_raw: str) -> tuple[dict, int]:
    """List users with optional role filter."""
    role = role_raw.strip().lower()
    role = role if role in ["student", "instructor", "admin"] else None
    users = fetch_users(role)
    return {"users": users}, 200


def list_user_courses(user_id: int) -> tuple[dict, int]:
    """List enrolled courses for one user."""
    courses = fetch_user_courses(user_id)
    return {"courses": courses}, 200


def enroll(user_id: int, data: dict) -> tuple[dict, int]:
    """Enroll a user into one course from payload."""
    data, error = require_dict(data)
    if error:
        return error, 400

    course_id = data.get("course_id")

    if not course_id:
        return {"error": "course_id is required"}, 400

    course_id, error = parse_optional_int(course_id, "course_id")
    if error or course_id is None:
        return {"error": "course_id must be an integer"}, 400

    if enroll_user(user_id, course_id):
        return {"status": "success", "message": "User enrolled successfully"}, 200
    return {"error": "Enrollment failed"}, 400


def list_courses() -> tuple[dict, int]:
    """List available courses."""
    try:
        courses = fetch_courses()
        return {"courses": courses}, 200
    except Exception as exc:
        return {"error": str(exc)}, 500
