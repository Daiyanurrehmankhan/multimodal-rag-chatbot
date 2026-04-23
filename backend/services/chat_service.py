"""Business logic for chat orchestration and streaming."""

import uuid

from backend.config import STAFF_ROLES
from backend.routes.helpers import (
    normalize_role,
    normalize_selected_course,
)
from backend.routes.state import chat_histories, last_responses
from backend.repositories.chat_repository import rebuild_history, save_session
from backend.schemas import parse_optional_int, require_dict
from backend.utils.identity import log_owner_resolution, resolve_owner_from_payload
from backend.utils.streaming import stream_chunks


def prepare_chat_request(data: dict) -> tuple[dict | None, tuple[dict, int] | None]:
    """Validate and normalize chat request payload."""
    data, error = require_dict(data)
    if error:
        return None, (error, 400)

    query = data.get("query")
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    user_role = data.get("user_role")
    normalized_role = normalize_role(user_role)
    selected_course = data.get("selected_course")
    if selected_course is None:
        selected_course = data.get("course_name")
    if selected_course is None:
        selected_course = data.get("course_id")

    if not query:
        return None, ({"error": "query is required"}, 400)

    normalized_course = normalize_selected_course(selected_course, user_role)

    if normalized_course is False:
        return None, ({"error": "selected_course must be a single course value, not a list."}, 400)

    if normalized_course is None:
        if normalized_role in STAFF_ROLES:
            normalized_course = "All Courses"
        else:
            return None, ({"error": "selected_course is required. Choose a course before chatting."}, 400)

    if normalized_role not in STAFF_ROLES and not normalized_course:
        return None, ({"error": "selected_course must be one of: AI/ML, IoT, Cyber Security, Data Science, Robotics, Web development, Cloud Computing"}, 400)

    if not session_id:
        session_id = str(uuid.uuid4())

    user_id, _ = parse_optional_int(user_id, "user_id")

    owner_key, owner_source = resolve_owner_from_payload(data)
    log_owner_resolution("/chat", owner_key, owner_source, user_id, user_role)

    prepared = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "user_role": user_role,
        "selected_course": normalized_course,
        "owner_key": owner_key,
    }
    return prepared, None


def start_chat_stream(prepared: dict, chat_callable=None):
    """Create stream generator for a chat turn and persist session metadata."""
    query = prepared["query"]
    session_id = prepared["session_id"]
    user_id = prepared["user_id"]
    user_role = prepared["user_role"]
    selected_course = prepared["selected_course"]
    owner_key = prepared["owner_key"]

    if session_id not in chat_histories:
        restored_history = rebuild_history(session_id)
        chat_histories[session_id] = restored_history

    save_session(
        session_id=session_id,
        owner_key=owner_key,
        user_id=user_id,
        user_role=user_role,
        selected_course=selected_course,
        title=query,
    )

    effective_chat = chat_callable or _default_chat_callable()

    def stream_with_context():
        collected = ""
        local_history = chat_histories[session_id]
        if not local_history:
            local_history.extend(rebuild_history(session_id))

        raw_stream = effective_chat(
            query,
            local_history,
            user_id=user_id,
            user_role=user_role,
            selected_course=selected_course,
            session_id=session_id,
            owner_key=owner_key,
        )

        for text in stream_chunks(raw_stream, include_done_marker=False):
            if text:
                collected += text
                yield text

        last_responses[session_id] = collected

    return stream_with_context()


def _default_chat_callable():
    """Lazily import chat callable to keep service import lightweight for tests."""
    from backend.services.chat_engine import chat as chat_callable

    return chat_callable
