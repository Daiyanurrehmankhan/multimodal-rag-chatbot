"""Centralized streaming pipeline utilities."""

from collections.abc import Iterable, Iterator

from backend.schemas import build_stream_chunk


def stream_chunks(
    raw_stream: Iterable[str],
    include_done_marker: bool = False,
    done_marker: str = "",
) -> Iterator[str]:
    """Yield formatted stream chunks with standardized error and completion handling."""
    try:
        for raw_chunk in raw_stream:
            dto = build_stream_chunk(content=raw_chunk or "")
            text = dto["content"]
            if text:
                yield text
    except Exception as exc:
        error_dto = build_stream_chunk(content="", error=str(exc))
        error_text = f"[stream_error] {error_dto['error']}"
        yield error_text
    finally:
        done_dto = build_stream_chunk(content=done_marker, done=True)
        if include_done_marker and done_dto["content"]:
            yield done_dto["content"]
