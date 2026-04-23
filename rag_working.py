"""Document ingestion, embeddings, and pgvector retrieval utilities for the RAG server."""

import os
import logging
import warnings
from dotenv import load_dotenv
from embeddings import get_embeddings
import psycopg2.extras
from backend.repositories.database import (
    CHAT_SESSIONS_TABLE_NAME,
    CHAT_TURNS_TABLE_NAME,
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    DOCUMENTS_TABLE_NAME,
    TABLE_NAME,
    create_chat_tables_if_not_exists,
    create_documents_table_if_not_exists,
    create_table_if_not_exists,
    get_db_connection,
    init_db_tables as init_core_db_tables,
)
from backend.repositories.user_data import (
    add_user_to_course,
    create_user,
    get_all_course_names,
    get_all_users,
    get_course_id_by_name,
    get_course_name_by_id,
    get_courses,
    get_user_by_id,
    get_user_courses,
    init_user_tables,
    verify_user,
)
from backend.services.document_ingestion import (
    infer_corpus_from_filename,
    load_and_chunk_single_file,
    load_and_prepare_documents,
    process_image_to_document,
)

# --------------------------------------------------
# LOGGING & WARNINGS
# --------------------------------------------------
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="`torch_dtype` is deprecated! Use `dtype` instead!"
)

load_dotenv()

embeddings = get_embeddings()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------


def init_db_tables():
    """Initialize all required database tables."""
    init_core_db_tables()
    init_user_tables()


def _build_chat_title(user_query: str) -> str:
    """Create a short display title for a chat session."""
    cleaned = " ".join(str(user_query or "").split()).strip()
    if not cleaned:
        return "New chat"
    # Keep titles compact for sidebar/list UIs.
    return cleaned[:60]


def upsert_chat_session(
    session_id: str,
    owner_key: str,
    user_id: int | None = None,
    user_role: str | None = None,
    selected_course: str | None = None,
    title: str | None = None,
) -> None:
    """Create or refresh a chat session row."""
    if not session_id or not owner_key:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            INSERT INTO {CHAT_SESSIONS_TABLE_NAME} (
                session_id, owner_key, user_id, user_role, selected_course, title, created_at, updated_at, last_message_at
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW(), NOW())
            ON CONFLICT (session_id)
            DO UPDATE SET
                owner_key = EXCLUDED.owner_key,
                -- Preserve prior non-null attributes when partial updates are sent.
                user_id = COALESCE(EXCLUDED.user_id, {CHAT_SESSIONS_TABLE_NAME}.user_id),
                user_role = COALESCE(EXCLUDED.user_role, {CHAT_SESSIONS_TABLE_NAME}.user_role),
                selected_course = COALESCE(EXCLUDED.selected_course, {CHAT_SESSIONS_TABLE_NAME}.selected_course),
                title = COALESCE(EXCLUDED.title, {CHAT_SESSIONS_TABLE_NAME}.title),
                updated_at = NOW(),
                last_message_at = NOW()
            """,
            (session_id, owner_key, user_id, user_role, selected_course, title),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def store_chat_turn(
    session_id: str,
    owner_key: str,
    turn_index: int,
    user_query: str,
    prompt_text: str,
    response_text: str,
    user_id: int | None = None,
    user_role: str | None = None,
    selected_course: str | None = None,
) -> int:
    """Persist one turn of chat and return the turn id."""
    if not session_id or not owner_key:
        return 0
    upsert_chat_session(
        session_id=session_id,
        owner_key=owner_key,
        user_id=user_id,
        user_role=user_role,
        selected_course=selected_course,
        title=_build_chat_title(user_query) if turn_index == 1 else None,
    )
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            INSERT INTO {CHAT_TURNS_TABLE_NAME} (
                session_id, turn_index, user_query, prompt_text, response_text, selected_course, user_role, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (session_id, turn_index)
            DO UPDATE SET
                user_query = EXCLUDED.user_query,
                prompt_text = EXCLUDED.prompt_text,
                response_text = EXCLUDED.response_text,
                selected_course = EXCLUDED.selected_course,
                user_role = EXCLUDED.user_role,
                updated_at = NOW()
            RETURNING id
            """,
            (session_id, turn_index, user_query, prompt_text, response_text, selected_course, user_role),
        )
        row = cur.fetchone()
        cur.execute(
            f"UPDATE {CHAT_SESSIONS_TABLE_NAME} SET updated_at = NOW(), last_message_at = NOW(), title = COALESCE(title, %s) WHERE session_id = %s",
            (_build_chat_title(user_query), session_id),
        )
        conn.commit()
        return row[0] if row else 0
    finally:
        cur.close()
        conn.close()


def update_chat_turn_response(session_id: str, turn_index: int, response_text: str) -> int:
    """Update the assistant response for a previously stored turn."""
    if not session_id:
        return 0
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            UPDATE {CHAT_TURNS_TABLE_NAME}
            SET response_text = %s, updated_at = NOW()
            WHERE session_id = %s AND turn_index = %s
            """,
            (response_text, session_id, turn_index),
        )
        conn.commit()
        return cur.rowcount
    finally:
        cur.close()
        conn.close()


def get_chat_sessions(owner_key: str | None = None, user_id: int | None = None) -> list[dict]:
    """Return saved chat sessions ordered by most recent activity."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        conditions = []
        params = []
        if owner_key:
            conditions.append("owner_key = %s")
            params.append(owner_key)
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cur.execute(
            f"""
            SELECT session_id, owner_key, user_id, user_role, selected_course, title, created_at, updated_at, last_message_at
            FROM {CHAT_SESSIONS_TABLE_NAME}
            {where_clause}
            ORDER BY last_message_at DESC, created_at DESC
            """,
            params,
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows] if rows else []
    finally:
        cur.close()
        conn.close()


def get_chat_turns(session_id: str) -> list[dict]:
    """Return stored chat turns for a session ordered from oldest to newest."""
    if not session_id:
        return []
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            f"""
            SELECT turn_index, user_query, prompt_text, response_text, selected_course, user_role, created_at, updated_at
            FROM {CHAT_TURNS_TABLE_NAME}
            WHERE session_id = %s
            ORDER BY turn_index ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows] if rows else []
    finally:
        cur.close()
        conn.close()


def delete_chat_session(session_id: str, owner_key: str | None = None) -> int:
    """Delete one chat session (and cascaded turns) by session id and optional owner key."""
    if not session_id:
        return 0
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if owner_key:
            cur.execute(
                f"DELETE FROM {CHAT_SESSIONS_TABLE_NAME} WHERE session_id = %s AND owner_key = %s",
                (session_id, owner_key),
            )
        else:
            cur.execute(
                f"DELETE FROM {CHAT_SESSIONS_TABLE_NAME} WHERE session_id = %s",
                (session_id,),
            )
        conn.commit()
        return cur.rowcount
    finally:
        cur.close()
        conn.close()


def rebuild_chat_history_from_turns(session_id: str) -> list[str]:
    """Recreate the in-memory Gemini history from stored chat turns."""
    history: list[str] = []
    for turn in get_chat_turns(session_id):
        prompt_text = turn.get("prompt_text")
        response_text = turn.get("response_text")
        if prompt_text:
            history.append(prompt_text)
        if response_text:
            history.append(response_text)
    return history


def index_single_document(
    file_path: str,
    course_id_or_name,
    initial_status: str = "pending",
    original_filename: str = None,
    short_description: str = None,
    generated_course_id: int | None = None,
) -> tuple[bool, str, int]:
    """
    Index one document for a course: chunk, embed, insert into document_chunks.
    Returns (success, message, document_id_or_0).
    original_filename: if provided, use this real filename in the database instead of the file_path basename.
    """
    create_table_if_not_exists()
    create_documents_table_if_not_exists()
    file_name = original_filename or os.path.basename(file_path)

    if isinstance(course_id_or_name, int):
        course_id = course_id_or_name
        course_name = get_course_name_by_id(course_id_or_name) or "General"
    else:
        course_name = str(course_id_or_name) if course_id_or_name else "General"
        course_id = get_course_id_by_name(course_name)

    if generated_course_id is not None:
        course_id = generated_course_id

    chunks = load_and_chunk_single_file(file_path)
    if not chunks:
        return False, "No content could be extracted from the file.", 0

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cleaned_description = str(short_description or "").strip()
        if not cleaned_description and chunks:
            cleaned_description = " ".join(chunks[0].page_content.split())[:300]

        document_id = 0
        try:
            cur.execute(
                f"""
                INSERT INTO {DOCUMENTS_TABLE_NAME} (file_name, course_id, course_name, short_description, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (file_name, course_name)
                DO UPDATE SET
                    course_id = EXCLUDED.course_id,
                    short_description = EXCLUDED.short_description,
                    source = EXCLUDED.source,
                    updated_at = NOW()
                RETURNING id
                """,
                (file_name, course_id, course_name, cleaned_description, file_path),
            )
            document_id_row = cur.fetchone()
            document_id = document_id_row[0] if document_id_row else 0
        except Exception:
            # Fallback for custom documents table schemas (e.g., document_name/description names).
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                """,
                (DOCUMENTS_TABLE_NAME,),
            )
            available_columns = {row[0] for row in cur.fetchall()}

            def first_existing(candidates: list[str]) -> str | None:
                for candidate in candidates:
                    if candidate in available_columns:
                        return candidate
                return None

            name_col = first_existing(["file_name", "document_name", "name"])
            course_name_col = first_existing(["course_name"])
            description_col = first_existing(["short_description", "description"])
            course_id_col = first_existing(["course_id"])

            if not name_col or not course_name_col:
                raise ValueError(
                    "documents table must include file_name/document_name and course_name columns"
                )

            column_names = [name_col, course_name_col]
            values = [file_name, course_name]

            if description_col:
                column_names.append(description_col)
                values.append(cleaned_description)

            if course_id_col:
                column_names.append(course_id_col)
                values.append(course_id)

            placeholders = ", ".join(["%s"] * len(column_names))
            columns_sql = ", ".join(column_names)
            cur.execute(
                f"INSERT INTO {DOCUMENTS_TABLE_NAME} ({columns_sql}) VALUES ({placeholders})",
                values,
            )

        for i, chunk in enumerate(chunks):
            embedding = embeddings.embed_query(chunk.page_content)
            cur.execute(
                f"INSERT INTO {TABLE_NAME} (course_name, file_name, source, chunk_index, content, status, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s::vector) "
                "ON CONFLICT (file_name, chunk_index) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, course_name = EXCLUDED.course_name, source = EXCLUDED.source, status = EXCLUDED.status",
                (course_name, file_name, file_path, i, chunk.page_content, initial_status, embedding),
            )
        conn.commit()
        cur.close()
        conn.close()
        return True, f"Indexed {len(chunks)} chunks for course {course_name}.", document_id
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        return False, str(e), 0


# --------------------------------------------------
# VECTORSTORE OPERATIONS (PGVECTOR)
# --------------------------------------------------
def create_initial_vectorstore():
    """Index every file under data/ into pgvector from scratch/upsert mode."""
    print("Creating initial pgvector store...")
    chunks = load_and_prepare_documents()
    conn = get_db_connection()
    cur = conn.cursor()
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk.page_content)
        course_name = chunk.metadata.get("course_name", "General")
        file_name = chunk.metadata.get("file_name")
        source = chunk.metadata.get("source")

        cur.execute(
            f"INSERT INTO {TABLE_NAME} (course_name, file_name, source, chunk_index, content, status, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s::vector) "
            "ON CONFLICT (file_name, chunk_index) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, course_name = EXCLUDED.course_name, source = EXCLUDED.source, status = EXCLUDED.status",
            (course_name, file_name, source, i, chunk.page_content, "approved", embedding)
        )
    conn.commit()
    cur.close()
    conn.close()
    print("pgvector store created.")
    return True

def load_vectorstore():
    """Placeholder for loading - pgvector doesn't need loading like Chroma."""
    return True

def perform_upsert():
    """Upsert newly loaded chunks into pgvector using file_name/chunk_index uniqueness."""
    print("Upserting new documents with embeddings...")
    chunks = load_and_prepare_documents()
    conn = get_db_connection()
    cur = conn.cursor()
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk.page_content)
        course_name = chunk.metadata.get("course_name", "General")
        file_name = chunk.metadata.get("file_name")
        source = chunk.metadata.get("source")

        cur.execute(
            f"INSERT INTO {TABLE_NAME} (course_name, file_name, source, chunk_index, content, status, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s::vector) "
            "ON CONFLICT (file_name, chunk_index) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, course_name = EXCLUDED.course_name, source = EXCLUDED.source, status = EXCLUDED.status",
            (course_name, file_name, source, i, chunk.page_content, "approved", embedding)
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Upserted {len(chunks)} chunks.")
    return True

# --------------------------------------------------
# INITIALIZATION (Check if table exists)
# --------------------------------------------------
def check_table_exists():
    create_table_if_not_exists()  # Ensure table exists
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (TABLE_NAME,))
    exists = cur.fetchone()[0]
    cur.close()
    conn.close()
    return exists

vectorstore_initialized = check_table_exists()

# --------------------------------------------------
# MAIN EXECUTION (Indexing / Upsert)
# --------------------------------------------------
if __name__ == "__main__":
    print("\n--- RAG INDEXING STARTED ---")
    if not vectorstore_initialized:
        create_initial_vectorstore()
    else:
        perform_upsert()
    print("--- RAG INDEXING COMPLETE ---")

# --------------------------------------------------
# ENROLLMENT (student's courses)
# --------------------------------------------------
def get_enrolled_course_ids(user_id: int) -> list[int]:
    """Return list of course_ids the user (student) is enrolled in. Uses enrollments table."""
    if user_id is None:
        return []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT course_id FROM enrollments WHERE student_id = %s",
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [r[0] for r in rows] if rows else []
    except Exception as e:
        logging.warning(f"get_enrolled_course_ids failed (enrollments table may not exist): {e}")
        return []


# --------------------------------------------------
# RETRIEVAL WITH REFERENCES (course-scoped)
# --------------------------------------------------
def get_response(query: str, allowed_course_names: list[str] | None = None):
    """
    Retrieve chunks from document_chunks, optionally restricted to the given course names.
    When allowed_course_names is empty/None, retrieval runs across all approved courses.
    """
    query_embedding = embeddings.embed_query(query)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    if allowed_course_names:
        # Student flow: query only the selected course(s).
        normalized_course_names = [str(c).strip().lower() for c in allowed_course_names if str(c).strip()]
        cur.execute(
            f"""
            SELECT content, embedding <=> %s::vector AS distance
            FROM {TABLE_NAME}
            WHERE LOWER(course_name) = ANY(%s)
              AND status = 'approved'
            ORDER BY distance
            LIMIT 5
            """,
            (query_embedding, normalized_course_names),
        )
    else:
        # Staff flow: query approved chunks across all courses.
        cur.execute(
            f"""
            SELECT content, embedding <=> %s::vector AS distance
            FROM {TABLE_NAME}
            WHERE status = 'approved'
            ORDER BY distance
            LIMIT 5
            """,
            (query_embedding,),
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return "[No documents are currently indexed for your courses. Do not use information from previous messages as document context.]"
    output = []
    for i, row in enumerate(rows, start=1):
        if not row:
            continue
        content = None
        if isinstance(row, dict):
            content = row.get('content')
        else:
            try:
                content = row[0]
            except Exception:
                content = None

        if not content:
            continue

        output.append(f"[Source {i}]\n{content}")

    if not output:
        return "[No documents are currently indexed for your selected course.]"

    return "\n\n---\n\n".join(output)


# --------------------------------------------------
# LIST DOCUMENTS (for debugging / UI)
# --------------------------------------------------
def list_document_names():
    """Return distinct file_name, source, course_name, status and uploaded_at from document_chunks."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        f"""
        SELECT file_name, source, course_name, MAX(status) AS status, MAX(created_at) AS uploaded_at
        FROM {TABLE_NAME}
        GROUP BY file_name, source, course_name
        ORDER BY MAX(created_at) DESC, file_name
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def list_documents_metadata():
    """Return document metadata rows from documents table with status inferred from document_chunks."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        f"""
        SELECT
            d.id AS document_id,
            d.file_name,
            d.course_id,
            d.course_name,
            d.short_description,
            d.source,
            d.created_at AS uploaded_at,
            COALESCE(chunk_status.status, 'approved') AS status
        FROM {DOCUMENTS_TABLE_NAME} d
        LEFT JOIN (
            SELECT
                file_name,
                course_name,
                CASE
                    WHEN BOOL_OR(status = 'rejected') THEN 'rejected'
                    WHEN BOOL_OR(status = 'pending') THEN 'pending'
                    ELSE 'approved'
                END AS status
            FROM {TABLE_NAME}
            GROUP BY file_name, course_name
        ) chunk_status
            ON chunk_status.file_name = d.file_name
           AND chunk_status.course_name = d.course_name
        ORDER BY d.created_at DESC, d.file_name
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def update_document_status(
    status: str,
    file_name: str | None = None,
    source: str | None = None,
) -> int:
    """Update status for all chunks of a document and return affected rows."""
    normalized = (status or "").strip().lower()
    if normalized not in ("pending", "approved", "rejected"):
        return 0

    if not file_name and not source:
        return 0

    conn = get_db_connection()
    cur = conn.cursor()
    conditions = []
    params = [normalized]

    if file_name:
        conditions.append("(file_name = %s OR file_name ILIKE %s)")
        params.extend([file_name, file_name])

    if source:
        conditions.append("(source = %s OR source ILIKE %s)")
        params.extend([source, source])

    query = f"UPDATE {TABLE_NAME} SET status = %s WHERE " + " AND ".join(conditions)
    try:
        cur.execute(query, params)
        conn.commit()
        updated = cur.rowcount
        cur.close()
        conn.close()
        return updated
    except Exception:
        conn.rollback()
        cur.close()
        conn.close()
        return 0


# --------------------------------------------------
# DELETION
# --------------------------------------------------
def delete_document(
    document_id: int | None = None,
    file_name: str | None = None,
    source: str | None = None,
    course_name: str | None = None,
) -> int:
    """
    Delete chunks for a given document from the database.
    Returns the number of rows deleted.
    Matches by file_name/source/course_name.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    conditions = []
    params = []

    if document_id is not None:
        cur.execute(
            f"SELECT file_name, source, course_name FROM {DOCUMENTS_TABLE_NAME} WHERE id = %s",
            (document_id,),
        )
        row = cur.fetchone()
        if row:
            file_name = file_name or row[0]
            source = source or row[1]
            course_name = course_name or row[2]

    if course_name:
        conditions.append("course_name = %s")
        params.append(course_name)
    if file_name:
        conditions.append("(file_name = %s OR file_name ILIKE %s OR source ILIKE %s)")
        params.extend([file_name, file_name, "%" + file_name + "%"])
    if source:
        conditions.append("(source = %s OR source ILIKE %s)")
        params.extend([source, "%" + source + "%"])

    if not conditions:
        cur.close()
        conn.close()
        return 0

    query = f"DELETE FROM {TABLE_NAME} WHERE " + " AND ".join(conditions)
    try:
        cur.execute(query, params)
        deleted_chunks = cur.rowcount

        metadata_conditions = []
        metadata_params = []
        if document_id is not None:
            metadata_conditions.append("id = %s")
            metadata_params.append(document_id)
        else:
            if file_name:
                metadata_conditions.append("file_name = %s")
                metadata_params.append(file_name)
            if course_name:
                metadata_conditions.append("course_name = %s")
                metadata_params.append(course_name)
            if source:
                metadata_conditions.append("source = %s")
                metadata_params.append(source)

        deleted_metadata = 0
        if metadata_conditions:
            metadata_query = f"DELETE FROM {DOCUMENTS_TABLE_NAME} WHERE " + " AND ".join(metadata_conditions)
            cur.execute(metadata_query, metadata_params)
            deleted_metadata = cur.rowcount

        conn.commit()
        deleted = deleted_chunks + deleted_metadata
        cur.close()
        conn.close()
        return deleted
    except Exception as e:
        print(f"Error deleting document from pgvector: {e}")
        conn.rollback()
        cur.close()
        conn.close()
        return 0
