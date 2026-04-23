"""Database connection and schema bootstrap helpers."""

import os

import psycopg2

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASS = os.getenv("DB_PASS", "pass")

TABLE_NAME = "document_chunks"
DOCUMENTS_TABLE_NAME = "documents"
CHAT_SESSIONS_TABLE_NAME = "chat_sessions"
CHAT_TURNS_TABLE_NAME = "chat_turns"


def get_db_connection():
    """Create a fresh PostgreSQL connection using environment configuration."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


def create_table_if_not_exists():
    """Ensure the pgvector chunk table exists with the expected schema."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            course_name TEXT,
            file_name TEXT,
            source TEXT,
            chunk_index INTEGER,
            content TEXT,
            status TEXT DEFAULT 'approved',
            embedding VECTOR(1024),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (file_name, chunk_index)
        );
    """)
    cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'approved';")
    cur.execute(f"UPDATE {TABLE_NAME} SET status = 'approved' WHERE status IS NULL OR TRIM(status) = '';")
    conn.commit()
    cur.close()
    conn.close()


def create_documents_table_if_not_exists():
    """Create document metadata table for uploaded documents."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                course_id INTEGER,
                course_name TEXT NOT NULL,
                short_description TEXT,
                source TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (file_name, course_name)
            );
        """)
        cur.execute(
            """
            SELECT conname
            FROM pg_constraint
            WHERE conrelid = %s::regclass
              AND contype = 'f'
            """,
            (DOCUMENTS_TABLE_NAME,),
        )
        for row in cur.fetchall():
            cur.execute(f"ALTER TABLE {DOCUMENTS_TABLE_NAME} DROP CONSTRAINT IF EXISTS {row[0]}")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE_NAME}_course_id ON {DOCUMENTS_TABLE_NAME}(course_id);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE_NAME}_course_name ON {DOCUMENTS_TABLE_NAME}(course_name);")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def create_chat_tables_if_not_exists():
    """Create tables used to persist chat sessions and chat turns."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {CHAT_SESSIONS_TABLE_NAME} (
                session_id TEXT PRIMARY KEY,
                owner_key TEXT NOT NULL,
                user_id INTEGER,
                user_role TEXT,
                selected_course TEXT,
                title TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                last_message_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {CHAT_TURNS_TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES {CHAT_SESSIONS_TABLE_NAME}(session_id) ON DELETE CASCADE,
                turn_index INTEGER NOT NULL,
                user_query TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                response_text TEXT,
                selected_course TEXT,
                user_role TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (session_id, turn_index)
            );
        """)
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{CHAT_SESSIONS_TABLE_NAME}_owner_key ON {CHAT_SESSIONS_TABLE_NAME}(owner_key);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{CHAT_SESSIONS_TABLE_NAME}_updated_at ON {CHAT_SESSIONS_TABLE_NAME}(updated_at DESC);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{CHAT_TURNS_TABLE_NAME}_session_id ON {CHAT_TURNS_TABLE_NAME}(session_id);")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def init_db_tables():
    """Initialize all required database tables."""
    create_table_if_not_exists()
    create_documents_table_if_not_exists()
    create_chat_tables_if_not_exists()
