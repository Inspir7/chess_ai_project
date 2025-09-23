import sqlite3

# Път до базата данни
db_path = r"C:\Users\prezi\PycharmProjects\chess_ai_project\data\datachess_games.db"

def remove_invalid_and_unknown_games():
    """Премахва партии с невалидни ходове и такива без разпознат дебют."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Изтриване на невалидните партии
    cursor.execute("DELETE FROM games WHERE eco = 'Invalid' OR opening = 'Invalid' OR eco = 'Unknown' OR opening = 'Unknown'")
    deleted_rows = cursor.rowcount
    conn.commit()
    conn.close()

    print(f"🗑️ Изтрити {deleted_rows} партии без дебют или с невалидни ходове.")

if __name__ == "__main__":
    remove_invalid_and_unknown_games()
