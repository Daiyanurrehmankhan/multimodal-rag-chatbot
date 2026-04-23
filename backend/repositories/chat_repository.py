"""Repository functions for chat/session persistence access."""

from rag_working import (
    delete_chat_session,
    get_chat_sessions,
    get_chat_turns,
    rebuild_chat_history_from_turns,
    upsert_chat_session,
)


def list_sessions(owner_key: str | None = None, user_id: int | None = None) -> list[dict]:
    """Fetch chat sessions by owner key and optional user id."""
    return get_chat_sessions(owner_key=owner_key, user_id=user_id)


def list_turns(session_id: str) -> list[dict]:
    """Fetch ordered turns for a chat session."""
    return get_chat_turns(session_id)


def rebuild_history(session_id: str) -> list[str]:
    """Rebuild in-memory model history from persisted turns."""
    return rebuild_chat_history_from_turns(session_id)


def save_session(
    session_id: str,
    owner_key: str,
    user_id: int | None = None,
    user_role: str | None = None,
    selected_course: str | None = None,
    title: str | None = None,
) -> None:
    """Upsert a chat session row."""
    upsert_chat_session(
        session_id=session_id,
        owner_key=owner_key,
        user_id=user_id,
        user_role=user_role,
        selected_course=selected_course,
        title=title,
    )


def delete_session(session_id: str, owner_key: str | None = None) -> int:
    """Delete one session by id and optional owner key scope."""
    return delete_chat_session(session_id=session_id, owner_key=owner_key)
