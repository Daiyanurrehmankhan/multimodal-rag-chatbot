"""Application bootstrap helpers."""

import logging
import os

from flask_cors import CORS

from backend.utils.observability import register_observability


REQUIRED_ENV_VARS = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASS",
    "GEMINI_API_KEY",
    "JINA_API_KEY",
)


def validate_required_environment():
    """Raise RuntimeError when required environment configuration is missing or invalid."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_csv}")

    db_port = os.getenv("DB_PORT", "").strip()
    try:
        parsed_port = int(db_port)
    except ValueError as exc:
        raise RuntimeError("DB_PORT must be an integer") from exc

    if parsed_port <= 0 or parsed_port > 65535:
        raise RuntimeError("DB_PORT must be between 1 and 65535")


def configure_app(app):
    """Attach common app configuration, CORS, and request logging hooks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.logger.setLevel(logging.INFO)

    configured_origins = os.getenv("FRONTEND_ORIGINS", "").strip()
    if configured_origins:
        allowed_origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    else:
        allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

    CORS(
        app,
        resources={r"/*": {"origins": allowed_origins}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    register_observability(app)


def initialize_runtime():
    """Prepare runtime directories and DB tables with fail-fast validation."""
    from backend.config import UPLOAD_FOLDER
    from rag_working import init_db_tables

    validate_required_environment()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    init_db_tables()
