"""Thin Flask app entrypoint for the RAG backend."""

from flask import Flask

from backend.bootstrap import configure_app, initialize_runtime
from backend.routes import register_routes


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    configure_app(app)
    initialize_runtime()
    register_routes(app)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=False, port=8000)
