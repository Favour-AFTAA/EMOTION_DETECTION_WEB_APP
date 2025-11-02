import sqlite3

conn = sqlite3.connect("database.db")
c = conn.cursor()

# Show all rows in users table
c.execute("SELECT * FROM users")
rows = c.fetchall()

for row in rows:
    print(row)

conn.close()
