"""Shared in-process state for streaming chat and exports."""

# In-memory cache used for active streaming conversations in this process.
chat_histories = {}

# Stores the last streamed assistant response per session for PDF export.
last_responses = {}
