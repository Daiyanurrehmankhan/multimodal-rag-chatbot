"""Chat streaming endpoints."""

from flask import Blueprint, Response, request, stream_with_context

from backend.services.chat_service import prepare_chat_request, start_chat_stream

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat_route():
    """Stream chat response for a single query turn."""
    data = request.json or {}
    prepared, error = prepare_chat_request(data)
    if error:
        payload, status = error
        return payload, status

    stream = start_chat_stream(prepared)

    return Response(
        stream_with_context(stream),
        mimetype="text/event-stream",
        headers={
            # Prevent proxy/runtime buffering so chunks reach the client immediately.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
