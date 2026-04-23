"""Route registration entrypoint."""

def register_routes(app):
    """Register all backend route blueprints."""
    from backend.routes.chat_routes import chat_bp
    from backend.routes.document_routes import document_bp
    from backend.routes.monolith import monolith_bp
    from backend.routes.session_routes import session_bp

    app.register_blueprint(monolith_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(session_bp)
    app.register_blueprint(document_bp)
