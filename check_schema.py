"""Quick utility to inspect the live schema of the users table."""

import os
import psycopg2


def main():
    """Print the current `users` table schema for quick DB diagnostics."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        user=os.getenv("DB_USER", "user"),
        password=os.getenv("DB_PASS", "pass"),
        dbname=os.getenv("DB_NAME", "rag_db"),
    )
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='users' ORDER BY ordinal_position"
        )
        print("Current users table schema:")
        for column_name, data_type in cur.fetchall():
            print(f"  {column_name}: {data_type}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
