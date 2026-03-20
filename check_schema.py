import psycopg2
conn = psycopg2.connect('host=localhost user=postgres password=aarcsol123 dbname=aarcsol_solutions')
cur = conn.cursor()
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='users' ORDER BY ordinal_position")
print("Current users table schema:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")
cur.close()
conn.close()
