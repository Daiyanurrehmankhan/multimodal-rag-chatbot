"""Business logic for chat session and history retrieval."""

from backend.repositories.chat_repository import delete_session, list_sessions, list_turns
from backend.schemas import build_delete_response, parse_optional_int, to_chat_turn_list, to_session_summary
from backend.utils.identity import log_owner_resolution, resolve_owner_from_query


def list_chat_sessions(query_args) -> tuple[dict, int]:
    """Return sessions for resolved owner identity and optional user id."""
    owner_key, owner_source = resolve_owner_from_query(query_args)
    user_id_raw = query_args.get("user_id")
    user_id, error = parse_optional_int(user_id_raw, "user_id")
    if error:
        return error, 400

    log_owner_resolution("/chat_sessions", owner_key, owner_source, user_id, None)

    sessions = [to_session_summary(row) for row in list_sessions(owner_key=owner_key or None, user_id=user_id)]
    return {"sessions": sessions}, 200


def get_chat_session_messages(session_id: str, query_args) -> tuple[dict, int]:
    """Return turns for a session with centralized owner guard."""
    owner_key, owner_source = resolve_owner_from_query(query_args)
    log_owner_resolution("/chat_sessions/<session_id>", owner_key, owner_source, None, None)

    if owner_key:
        sessions = list_sessions(owner_key=owner_key)
        if not any(session["session_id"] == session_id for session in sessions):
            return {"error": "session not found"}, 404

    turns = to_chat_turn_list(list_turns(session_id))
    if not turns:
        return {"session_id": session_id, "turns": []}, 200
    return {"session_id": session_id, "turns": turns}, 200


def delete_chat_session(session_id: str, query_args) -> tuple[dict, int]:
    """Delete one chat session constrained by resolved owner identity."""
    owner_key, owner_source = resolve_owner_from_query(query_args)
    log_owner_resolution("/chat_sessions/<session_id>", owner_key, owner_source, None, None)

    deleted_count = delete_session(session_id=session_id, owner_key=owner_key)
    response = build_delete_response(deleted_count)
    if response["deleted_count"] == 0:
        return response, 404
    return response, 200
