"""Shared DTOs and normalization helpers for backend responses."""

from typing import Any, TypedDict


class SessionSummaryDTO(TypedDict):
    session_id: str
    owner_key: str
    user_id: int | None
    user_role: str | None
    selected_course: str | None
    title: str | None
    created_at: Any
    updated_at: Any
    last_message_at: Any


class ChatTurnDTO(TypedDict):
    turn_index: int
    user_query: str
    prompt_text: str
    response_text: str | None
    selected_course: str | None
    user_role: str | None
    created_at: Any
    updated_at: Any


class DeleteResponseDTO(TypedDict):
    status: str
    deleted_count: int


class UpdateResponseDTO(TypedDict):
    status: str
    updated_count: int


class StreamChunkDTO(TypedDict):
    type: str
    content: str
    done: bool
    error: str | None


def to_session_summary(row: dict) -> SessionSummaryDTO:
    """Normalize a session row into a consistent response contract."""
    return {
        "session_id": str(row.get("session_id") or ""),
        "owner_key": str(row.get("owner_key") or ""),
        "user_id": row.get("user_id"),
        "user_role": row.get("user_role"),
        "selected_course": row.get("selected_course"),
        "title": row.get("title"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "last_message_at": row.get("last_message_at"),
    }


def to_chat_turn(row: dict) -> ChatTurnDTO:
    """Normalize a chat turn row into a consistent response contract."""
    return {
        "turn_index": int(row.get("turn_index") or 0),
        "user_query": str(row.get("user_query") or ""),
        "prompt_text": str(row.get("prompt_text") or ""),
        "response_text": row.get("response_text"),
        "selected_course": row.get("selected_course"),
        "user_role": row.get("user_role"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def to_chat_turn_list(rows: list[dict]) -> list[ChatTurnDTO]:
    """Normalize ordered chat turn rows."""
    return [to_chat_turn(row) for row in rows]


def build_delete_response(deleted_count: int) -> DeleteResponseDTO:
    """Build a standardized delete response payload."""
    if deleted_count <= 0:
        return {"status": "not_found_or_error", "deleted_count": 0}
    return {"status": "deleted", "deleted_count": int(deleted_count)}


def build_update_response(updated_count: int) -> UpdateResponseDTO:
    """Build a standardized update response payload."""
    if updated_count <= 0:
        return {"status": "not_found_or_error", "updated_count": 0}
    return {"status": "updated", "updated_count": int(updated_count)}


def build_stream_chunk(content: str, done: bool = False, error: str | None = None) -> StreamChunkDTO:
    """Build a stream chunk contract for chat streaming utilities."""
    chunk_type = "error" if error else ("done" if done else "chunk")
    return {
        "type": chunk_type,
        "content": str(content or ""),
        "done": bool(done),
        "error": error,
    }
