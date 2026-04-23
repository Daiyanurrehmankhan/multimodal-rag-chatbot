# RAG Chatbot Backend

Flask-based RAG chatbot backend with layered route, service, repository, schema, and utility modules.

## What changed

The backend has been split by responsibility:

- HTTP routes live in `backend/routes/`
- business logic lives in `backend/services/`
- data access wrappers live in `backend/repositories/`
- DTO and validation helpers live in `backend/schemas/`
- shared identity, streaming, and observability helpers live in `backend/utils/`

The app entrypoint is `rag_server.py`, which builds the Flask app, configures startup behavior, and registers the blueprints.

## Project Layout

- `rag_server.py` - Flask app factory and startup entrypoint
- `app.py` - Gemini chat orchestration and streamed response generation
- `rag_working.py` - legacy data, retrieval, and persistence layer used by repositories
- `backend/`
  - `routes/` - blueprint modules for chat, sessions, documents, and shared route helpers
  - `services/` - request validation, business logic, streaming orchestration, and PDF export
  - `repositories/` - database and persistence wrappers
  - `schemas/` - DTO builders and request validation helpers
  - `utils/` - owner resolution, streaming helpers, and request logging/error handling
- `tests/` - unit and route integration tests

## Requirements

Install the dependencies listed in `requirements.txt`.

## Local Setup

Create and activate the virtual environment if you do not already have one:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set any required environment variables before running the app. The chat flow expects a Gemini API key when you use live model calls.

Use `.env.example` as the starter template for local or production environment variables.

## Environment Variables

The app reads configuration from `.env` or the process environment.

Startup validates required values and fails fast if any are missing or invalid.

- `GEMINI_API_KEY` - required for live chat and image/Gemini calls
- `GEMINI_CHAT_MODELS` - optional comma-separated fallback order for chat models
- `JINA_API_KEY` - required for Jina embedding generation
- `JINA_EMBEDDING_MODEL` - optional embedding model name, defaults to `jina-embeddings-v3`
- `DB_HOST` - required PostgreSQL host
- `DB_PORT` - required PostgreSQL port (must be an integer between 1 and 65535)
- `DB_NAME` - required PostgreSQL database name
- `DB_USER` - required PostgreSQL username
- `DB_PASS` - required PostgreSQL password
- `FRONTEND_ORIGINS` - optional comma-separated allowed CORS origins (default: `http://localhost:5173,http://127.0.0.1:5173`)

## Run the App

Start the Flask app from the repository root:

```powershell
venv\Scripts\python.exe rag_server.py
```

The server runs on port `8000` by default.

## Tests

Run the full test suite with:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite includes:

- service-level tests for chat, sessions, identity, PDF export, and streaming helpers
- repository delegation tests
- Flask route integration tests using the app factory and test client

## Deployment

See `DEPLOYMENT.md` for end-to-end backend + PostgreSQL deployment steps, including Windows service and Linux systemd guidance.

## API Endpoints

- `POST /chat` - stream a chat response for one turn
- `GET /chat_sessions` - list chat sessions for the resolved owner scope
- `GET /chat_sessions/<session_id>` - fetch stored turns for one session
- `DELETE /chat_sessions/<session_id>` - delete one stored session
- `POST /upload_document` - upload and index a document
- `GET /list_documents` - list stored document metadata
- `POST /set_document_status` - update document status by selector
- `POST /approve_document` - mark a document as approved
- `POST /reject_document` - mark a document as rejected
- `POST /delete_document` - delete a document by selector
- `GET /download_pdf?session_id=...&turn_index=...` - download one session response as PDF (`turn_index` is canonical)
- `GET /health` - health check endpoint (returns `{ "status": "ok" }`)

## CORS Verification Example

To verify CORS headers are set correctly, run:

```bash
curl -i -X OPTIONS http://localhost:8000/health \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: GET"
```

You should see `Access-Control-Allow-Origin: http://localhost:5173` in the response headers if CORS is configured properly.

## Notes

- Route registration uses lazy imports to avoid circular imports during testing.
- Route integration tests patch startup table initialization so they do not depend on live database setup.
- The PDF export endpoint accepts snake_case selectors only (`turn_index`, temporary compatibility: `message_index`, `response_text`).
- `backend/services/chat_service.py` is intentionally testable through dependency injection of the chat callable.
- `backend/utils/observability.py` adds request/response logging and shared error handling across blueprints.