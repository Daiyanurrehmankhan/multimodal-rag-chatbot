"""Business logic dedicated to PDF export formatting and generation."""

import re
from html import escape
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from backend.repositories.chat_repository import list_turns
from backend.routes.state import last_responses


def _markdown_to_story(text: str):
    """Convert markdown-like text into ReportLab story elements."""
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    h1_style = styles.get("Heading1", normal_style)
    h2_style = styles.get("Heading2", normal_style)
    h3_style = styles.get("Heading3", normal_style)

    story = []
    bullet_items = []

    def flush_bullets():
        nonlocal bullet_items
        if bullet_items:
            list_items = [
                ListItem(Paragraph(item, normal_style)) for item in bullet_items
            ]
            story.append(ListFlowable(list_items, bulletType="bullet"))
            story.append(Spacer(1, 6))
            bullet_items = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            flush_bullets()
            story.append(Spacer(1, 8))
            continue

        stripped = line.lstrip()

        if stripped.startswith("### "):
            flush_bullets()
            heading_text = stripped[4:]
            safe = escape(heading_text)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            story.append(Paragraph(safe, h3_style))
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("## "):
            flush_bullets()
            heading_text = stripped[3:]
            safe = escape(heading_text)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            story.append(Paragraph(safe, h2_style))
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("# "):
            flush_bullets()
            heading_text = stripped[2:]
            safe = escape(heading_text)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            story.append(Paragraph(safe, h1_style))
            story.append(Spacer(1, 10))
            continue

        is_bullet = stripped.startswith("* ") or stripped.startswith("- ")
        if is_bullet:
            item_text = stripped[2:]
            safe = escape(item_text)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            bullet_items.append(safe)
        else:
            flush_bullets()
            safe = escape(line)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            para = Paragraph(safe, normal_style)
            story.append(para)
            story.append(Spacer(1, 4))

    flush_bullets()
    return story


def _resolve_response_text(
    session_id: str,
    turn_index: str | int | None,
    message_index: str | int | None,
    response_text: str | None,
):
    """Resolve response text using canonical turn_index then compatibility selectors."""

    resolved_turn_index = None
    if turn_index not in (None, ""):
        try:
            resolved_turn_index = int(turn_index)
        except (TypeError, ValueError):
            return None, ({"error": "turn_index must be an integer"}, 400)

    if resolved_turn_index is not None:
        turns = list_turns(session_id)
        for turn in turns:
            if int(turn.get("turn_index") or 0) == resolved_turn_index:
                response_text = turn.get("response_text")
                if response_text:
                    return response_text, None
                return None, ({"error": "No response text found for the requested turn"}, 404)
        return None, ({"error": "Requested turn was not found for this session"}, 404)

    resolved_message_index = None
    if message_index not in (None, ""):
        try:
            resolved_message_index = int(message_index)
        except (TypeError, ValueError):
            return None, ({"error": "message_index must be an integer"}, 400)
        turns = list_turns(session_id)
        response_candidates = [turn.get("response_text") for turn in turns if turn.get("response_text")]
        if resolved_message_index < 0 or resolved_message_index >= len(response_candidates):
            return None, ({"error": "Requested message index is out of range for this session"}, 404)
        return response_candidates[resolved_message_index], None

    if response_text and str(response_text).strip():
        return str(response_text), None

    turns = list_turns(session_id)

    for turn in reversed(turns):
        response_text = turn.get("response_text")
        if response_text:
            return response_text, None

    cached_response = last_responses.get(session_id)
    if cached_response:
        return cached_response, None

    return None, ({"error": "No response available for this session"}, 404)


def build_pdf_response(
    session_id: str,
    turn_index: str | int | None = None,
    message_index: str | int | None = None,
    response_text: str | None = None,
):
    """Build in-memory PDF for a chat session response."""
    if not session_id:
        return None, ({"error": "session_id is required"}, 400)

    response_text, resolve_error = _resolve_response_text(
        session_id=session_id,
        turn_index=turn_index,
        message_index=message_index,
        response_text=response_text,
    )
    if resolve_error:
        return None, resolve_error

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story = _markdown_to_story(response_text)
    doc.build(story)
    buffer.seek(0)
    return buffer, None
