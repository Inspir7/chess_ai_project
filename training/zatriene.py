import sqlite3

conn = sqlite3.connect("/home/presi/projects/chess_ai_project/data/datachess_games.db")
cursor = conn.cursor()

# Виж структурата на таблицата
cursor.execute("PRAGMA table_info(games)")
columns = cursor.fetchall()

print("Колони в базата данни:")
for col in columns:
    print(col)

conn.close()