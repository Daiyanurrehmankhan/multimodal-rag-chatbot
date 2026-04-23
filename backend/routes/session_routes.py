"""Chat history and session endpoints."""

from flask import Blueprint, request

from backend.services.session_service import (
    delete_chat_session,
    get_chat_session_messages,
    list_chat_sessions,
)

session_bp = Blueprint("sessions", __name__)


@session_bp.route("/chat_sessions", methods=["GET"])
def chat_sessions_route():
    """Return saved chat sessions for the current browser/user."""
    return list_chat_sessions(request.args)


@session_bp.route("/chat_sessions/<session_id>", methods=["GET"])
def chat_session_messages_route(session_id):
    """Return stored turns for a chat session."""
    return get_chat_session_messages(session_id=session_id, query_args=request.args)


@session_bp.route("/chat_sessions/<session_id>", methods=["DELETE"])
def delete_chat_session_route(session_id):
    """Delete one stored chat session for the resolved owner scope."""
    return delete_chat_session(session_id=session_id, query_args=request.args)
