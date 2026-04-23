"""General, auth, and user/course routes.

Domain-specific endpoints are split into dedicated modules under backend.routes.
"""

from flask import Blueprint, render_template, request

from backend.services.user_service import (
    enroll,
    get_user,
    list_courses,
    list_user_courses,
    list_users,
    login,
    register,
)


monolith_bp = Blueprint("monolith", __name__)


# Simple health check endpoint
@monolith_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for uptime monitoring."""
    return {"status": "ok"}, 200


@monolith_bp.route("/")
def index():
    return render_template("index.html")


@monolith_bp.route("/login", methods=["POST"])
def login_route():
    return login(request.json or {})


@monolith_bp.route("/register", methods=["POST"])
def register_route():
    return register(request.json or {})


@monolith_bp.route("/user/<int:user_id>", methods=["GET"])
def get_user_route(user_id):
    return get_user(user_id)


@monolith_bp.route("/users", methods=["GET"])
def get_users_route():
    return list_users(request.args.get("role", ""))


@monolith_bp.route("/user/<int:user_id>/courses", methods=["GET"])
def get_user_courses_route(user_id):
    return list_user_courses(user_id)


@monolith_bp.route("/user/<int:user_id>/enroll", methods=["POST"])
def enroll_user_route(user_id):
    return enroll(user_id, request.json or {})


@monolith_bp.route("/courses", methods=["GET"])
def courses_route():
    return list_courses()
