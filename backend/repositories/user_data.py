"""User, course, and enrollment persistence helpers."""

import hashlib
import logging

import psycopg2
import psycopg2.extras

from backend.repositories.database import TABLE_NAME, get_db_connection


def create_users_table_if_not_exists():
    """Create users table if it does not exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
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
            """
        )
        cur.execute("""ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT DEFAULT '';""")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def create_courses_table_if_not_exists():
    """Create courses table if it does not exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS courses (
                id SERIAL PRIMARY KEY,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_courses_code ON courses(code);")
        cur.execute("INSERT INTO courses (code, name) VALUES ('AIML', 'AI/ML') ON CONFLICT (code) DO NOTHING;")
        cur.execute("INSERT INTO courses (code, name) VALUES ('IOT', 'IoT') ON CONFLICT (code) DO NOTHING;")
        cur.execute("INSERT INTO courses (code, name) VALUES ('CYBER', 'Cyber Security') ON CONFLICT (code) DO NOTHING;")
        cur.execute("INSERT INTO courses (code, name) VALUES ('DATA', 'Data Science') ON CONFLICT (code) DO NOTHING;")
        cur.execute("INSERT INTO courses (code, name) VALUES ('ROBOTICS', 'Robotics') ON CONFLICT (code) DO NOTHING;")
        cur.execute("INSERT INTO courses (code, name) VALUES ('WEB', 'Web development') ON CONFLICT (code) DO NOTHING;")
        cur.execute("INSERT INTO courses (code, name) VALUES ('CLOUD', 'Cloud Computing') ON CONFLICT (code) DO NOTHING;")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def create_enrollments_table_if_not_exists():
    """Create enrollments table if it does not exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS enrollments (
                id SERIAL PRIMARY KEY,
                student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                enrolled_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (student_id, course_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_student_id ON enrollments(student_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_course_id ON enrollments(course_id);")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def init_user_tables():
    """Initialize user, course, and enrollment tables."""
    create_users_table_if_not_exists()
    create_courses_table_if_not_exists()
    create_enrollments_table_if_not_exists()


def _hash_password(password: str) -> str:
    return hashlib.sha256((password or "").encode("utf-8")).hexdigest()


def _is_valid_password(password: str, stored_password: str) -> bool:
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
            (email,),
        )
        user = cur.fetchone()
        if user and _is_valid_password(password, user["password_hash"]):
            return {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
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
            (name, email, password_hash, role),
        )
        user = cur.fetchone()
        conn.commit()
        if user:
            return {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
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
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
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
    except Exception as exc:
        logging.warning(f"Failed to add user to course: {exc}")
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
        cur.execute(
            """
            SELECT c.id, c.code, c.name
            FROM courses c
            INNER JOIN enrollments e ON c.id = e.course_id
            WHERE e.student_id = %s
            ORDER BY c.name
            """,
            (user_id,),
        )
        courses = cur.fetchall()
        return [dict(row) for row in courses] if courses else []
    finally:
        cur.close()
        conn.close()


def get_courses() -> list[dict]:
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
