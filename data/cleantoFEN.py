import sqlite3
import pandas as pd
import chess
import chess.pgn

# Свързване с базата данни
conn = sqlite3.connect("datachess_games.db")
cursor = conn.cursor()

print("Добавяне на FEN колона, ако не съществува...")
# Добавяне на FEN колона, ако не съществува
try:
    cursor.execute("ALTER TABLE games ADD COLUMN fen TEXT")
    conn.commit()
except sqlite3.OperationalError:
    print("Колоната FEN вече съществува.")

print("Извличане на FEN от наличните SAN/UCI ходове...")
# Извличане на FEN от наличните SAN/UCI ходове
cursor.execute("SELECT id, moves_san FROM games WHERE fen IS NULL OR fen = ''")
games = cursor.fetchall()
print(f"Общо партии за обработка: {len(games)}")

for game_id, moves_san in games:
    board = chess.Board()
    if moves_san:
        try:
            for move in moves_san.split():
                board.push_san(move)
            fen_position = board.fen()
            cursor.execute("UPDATE games SET fen = ? WHERE id = ?", (fen_position, game_id))
        except ValueError:
            print(f"Грешка при обработка на партия ID {game_id}")
conn.commit()

print("Премахване на невалидни партии...")
# Премахване на невалидни партии (без ходове, резултат или FEN)
cursor.execute("DELETE FROM games WHERE moves_san IS NULL OR moves_san = '' OR result IS NULL OR result = '' OR fen IS NULL OR fen = ''")
conn.commit()

print("Премахване на дублиращи се записи...")
# Премахване на дублиращи се записи
cursor.execute("CREATE TABLE temp_games AS SELECT DISTINCT * FROM games")
cursor.execute("DROP TABLE games")
cursor.execute("ALTER TABLE temp_games RENAME TO games")
conn.commit()

print("Попълване на липсващи стойности по подразбиране...")
# Проверка за липсващи полета и попълване със стойности по подразбиране
cursor.execute("UPDATE games SET num_moves = 0 WHERE num_moves IS NULL")
conn.commit()

# Преброяване на почистените игри
cursor.execute("SELECT COUNT(*) FROM games")
game_count = cursor.fetchone()[0]
print(f"Оставащи игри след почистване: {game_count}")

# Затваряне на връзката с базата
conn.close()
print("Процесът на почистване е завършен.")
