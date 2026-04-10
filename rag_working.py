"""Document ingestion, embeddings, and pgvector retrieval utilities for the RAG server."""

import os
import logging
import warnings
import hashlib
import base64
from glob import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from google import genai
from langchain.docstore.document import Document
from embeddings import get_embeddings
import psycopg2
import psycopg2.extras

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
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASS = os.getenv("DB_PASS", "pass")
TABLE_NAME = "document_chunks"
DOCUMENTS_TABLE_NAME = "documents"

# --------------------------------------------------
# GEMINI CLIENT (IMAGES)
# --------------------------------------------------
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    print("Warning: Gemini Client not initialized.")
    gemini_client = None

# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------
def get_db_connection():
    """Create a fresh PostgreSQL connection using environment configuration."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
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
    cur.execute(
        f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'approved';"
    )
    cur.execute(
        f"UPDATE {TABLE_NAME} SET status = 'approved' WHERE status IS NULL OR TRIM(status) = '';"
    )
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
            cur.execute(
                f"ALTER TABLE {DOCUMENTS_TABLE_NAME} DROP CONSTRAINT IF EXISTS {row[0]}"
            )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE_NAME}_course_id ON {DOCUMENTS_TABLE_NAME}(course_id);"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE_NAME}_course_name ON {DOCUMENTS_TABLE_NAME}(course_name);"
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def infer_corpus_from_filename(filename: str) -> str:
    """
    Lightweight, domain-agnostic corpus label based on filename.
    Currently returns a generic label; adjust if you want custom grouping.
    """
    return "General"


def _extract_text_from_parts(response_obj) -> str:
    """Return concatenated text-only parts from a Gemini response object."""
    texts = []
    candidates = getattr(response_obj, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                texts.append(part_text)
    return "".join(texts)

def process_image_to_document(file_path: str, client: genai.Client):
    """Generate a text summary for images so they can be embedded like text docs."""
    if not client:
        return []
    try:
        with open(file_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        mime_type = f"image/{os.path.splitext(file_path)[1].lstrip('.')}"
        prompt = (
            "Provide a concise, detailed, professional summary of the image content. "
            "Focus on technical or informational aspects only."
        )
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                {"role": "user", "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": base64_image}}
                ]}
            ]
        )
        response_text = _extract_text_from_parts(response)
        if not response_text.strip():
            response_text = "No descriptive text could be extracted from this image."
        return [
            Document(
                page_content=response_text,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "type": "image_description",
                    "corpus": "Images"
                }
            )
        ]
    except Exception as e:
        print(f"Image processing error: {e}")
        return []

def load_and_prepare_documents():
    """Load supported files from data/, split content, and return clean chunk documents."""
    all_files = glob("data/*")
    documents = []
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        loader = None
        if ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
            print(f"Loaded PDF: {file_path}")
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            documents.extend(process_image_to_document(file_path, gemini_client))
            continue
        if not loader:
            continue
        try:
            loaded_docs = loader.load()
            for doc in loaded_docs:
                if not doc.page_content.strip():
                    continue
                source_path = doc.metadata.get("source", file_path)
                file_name = os.path.basename(source_path)
                doc.metadata["file_name"] = file_name
                doc.metadata["corpus"] = infer_corpus_from_filename(file_name)
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75
    )
    chunks = splitter.split_documents(documents)
    clean_chunks = [c for c in chunks if c.page_content and len(c.page_content.strip()) >= 30]

    for chunk in clean_chunks:
        source_path = chunk.metadata.get("source", "unknown")
        file_name = chunk.metadata.get("file_name", os.path.basename(source_path))
        chunk.metadata["file_name"] = file_name
        chunk.metadata["source"] = source_path

    return clean_chunks


def load_and_chunk_single_file(file_path: str):
    """Load one file and return list of chunk documents with metadata (file_name, source, etc.)."""
    ext = os.path.splitext(file_path)[1].lower()
    documents = []
    if ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
        for doc in loader.load():
            if doc.page_content.strip():
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["source"] = file_path
                documents.append(doc)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
        for doc in loader.load():
            if doc.page_content.strip():
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["source"] = file_path
                documents.append(doc)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        for doc in loader.load():
            if doc.page_content.strip():
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["source"] = file_path
                documents.append(doc)
    elif ext in [".jpg", ".jpeg", ".png"] and gemini_client:
        documents = process_image_to_document(file_path, gemini_client)
    else:
        return []
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if c.page_content and len(c.page_content.strip()) >= 30]


# --------------------------------------------------
# USER & ENROLLMENT FUNCTIONS
# --------------------------------------------------
def create_users_table_if_not_exists():
    """Create users table if it doesn't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('student', 'instructor', 'admin')),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # Add password_hash column if it doesn't exist (for existing tables)
        cur.execute("""
            ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT DEFAULT '';
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def create_courses_table_if_not_exists():
    """Create courses table if it doesn't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id SERIAL PRIMARY KEY,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_courses_code ON courses(code);")
        # Insert default courses if they don't exist
        cur.execute("INSERT INTO courses (code, name) VALUES ('LIFE', 'Life Insurance') ON CONFLICT (code) DO NOTHING;")
        cur.execute("INSERT INTO courses (code, name) VALUES ('ISLAMIC', 'Islamic Banking') ON CONFLICT (code) DO NOTHING;")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def create_enrollments_table_if_not_exists():
    """Create enrollments table if it doesn't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                id SERIAL PRIMARY KEY,
                student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                enrolled_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (student_id, course_id)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_student_id ON enrollments(student_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_course_id ON enrollments(course_id);")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def init_db_tables():
    """Initialize all required database tables."""
    create_table_if_not_exists()
    create_users_table_if_not_exists()
    create_courses_table_if_not_exists()
    create_enrollments_table_if_not_exists()
    create_documents_table_if_not_exists()


def _hash_password(password: str) -> str:
    """Create deterministic SHA-256 hash for password storage."""
    return hashlib.sha256((password or "").encode("utf-8")).hexdigest()


def _is_valid_password(password: str, stored_password: str) -> bool:
    """Validate password against stored hash (supports legacy plain-text rows)."""
    if not stored_password:
        return False
    provided_hash = _hash_password(password)
    return stored_password == provided_hash or stored_password == (password or "")


def verify_user(email: str, password: str) -> dict | None:
    """Verify user credentials and return user object if valid."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute(
            "SELECT id, name, email, role, is_active, password_hash FROM users WHERE email = %s AND is_active = TRUE",
            (email,)
        )
        user = cur.fetchone()
        if user and _is_valid_password(password, user["password_hash"]):
            return {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
                'role': user['role']
            }
        return None
    finally:
        cur.close()
        conn.close()


def create_user(name: str, email: str, password: str, role: str) -> dict | None:
    """Create a new user."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        password_hash = _hash_password(password)
        cur.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, %s) RETURNING id, name, email, role",
            (name, email, password_hash, role)
        )
        user = cur.fetchone()
        conn.commit()
        if user:
            return {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
                'role': user['role']
            }
        return None
    except psycopg2.IntegrityError:
        conn.rollback()
        return None
    finally:
        cur.close()
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    """Get user by ID."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("SELECT id, name, email, role, is_active FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        if user:
            return {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
                'role': user['role']
            }
        return None
    finally:
        cur.close()
        conn.close()


def get_all_users(role: str | None = None) -> list[dict]:
    """Get all users, optionally filtered by role."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        if role:
            cur.execute("SELECT id, name, email, role FROM users WHERE role = %s AND is_active = TRUE ORDER BY name", (role,))
        else:
            cur.execute("SELECT id, name, email, role FROM users WHERE is_active = TRUE ORDER BY name")
        users = cur.fetchall()
        return [dict(row) for row in users] if users else []
    finally:
        cur.close()
        conn.close()


def add_user_to_course(user_id: int, course_id: int) -> bool:
    """Enroll a user in a course."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO enrollments (student_id, course_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (user_id, course_id))
        conn.commit()
        return cur.rowcount > 0
    except Exception as e:
        logging.warning(f"Failed to add user to course: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def get_user_courses(user_id: int) -> list[dict]:
    """Get all courses for a user."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT c.id, c.code, c.name
            FROM courses c
            INNER JOIN enrollments e ON c.id = e.course_id
            WHERE e.student_id = %s
            ORDER BY c.name
        """, (user_id,))
        courses = cur.fetchall()
        return [dict(row) for row in courses] if courses else []
    finally:
        cur.close()
        conn.close()


def get_courses():
    """Return course list from database."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("SELECT id, code, name FROM courses ORDER BY name")
        courses = cur.fetchall()
        return [dict(row) for row in courses] if courses else []
    finally:
        cur.close()
        conn.close()


def get_course_name_by_id(course_id: int) -> str | None:
    """Return course name for a course id, or None when not found."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM courses WHERE id = %s", (course_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()
        conn.close()


def get_course_id_by_name(course_name: str) -> int | None:
    """Return course id for a course name, case-insensitive, or None when not found."""
    if not course_name:
        return None
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM courses WHERE LOWER(name) = LOWER(%s) LIMIT 1", (course_name,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()
        conn.close()


def get_all_course_names() -> list[str]:
    """Return all distinct indexed course names from document_chunks."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT DISTINCT course_name FROM {TABLE_NAME} WHERE course_name IS NOT NULL AND TRIM(course_name) <> '' ORDER BY course_name"
        )
        rows = cur.fetchall()
        return [row[0] for row in rows if row and row[0]]
    finally:
        cur.close()
        conn.close()


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
    If allowed_course_names is None or empty, no documents are returned (student must select a course).
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
