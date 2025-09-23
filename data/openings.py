import sqlite3
import chess
import chess.pgn
import pandas as pd
import time
import io

# Пътища до файловете
openings_csv_path = r"C:\Users\prezi\PycharmProjects\chess_ai_project\data\chess_openings.csv"
db_path = r"C:\Users\prezi\PycharmProjects\chess_ai_project\data\datachess_games.db"
invalid_games_path = r"C:\Users\prezi\PycharmProjects\chess_ai_project\data\invalid_games.txt"

# Зареждане на дебютите
openings_df = pd.read_csv(openings_csv_path)


def san_to_uci(san_moves):
    """Конвертира SAN ходове (като 'e4 e5 Nf3 Nc6') в UCI (като 'e2e4 e7e5 g1f3 b8c6')."""
    board = chess.Board()
    uci_moves = []

    for san in san_moves.split():
        try:
            move = board.parse_san(san)
            uci_moves.append(move.uci())
            board.push(move)
        except ValueError:
            return "Invalid"  # Ако има грешка, маркираме хода като невалиден

    return " ".join(uci_moves)


# Конвертиране на дебютите в UCI (ако все още не са)
if 'uci_moves' not in openings_df.columns:
    openings_df['uci_moves'] = openings_df['moves'].apply(san_to_uci)


def get_opening_from_moves(moves_uci):
    """Опитва се да разпознае дебюта на базата на първите ходове."""
    if moves_uci == "Invalid":
        return "Invalid", "Invalid"

    board = chess.Board()
    moves = moves_uci.split()

    for move in moves:
        try:
            board.push_uci(move)
        except ValueError:
            with open(invalid_games_path, "a") as f:
                f.write(moves_uci + "\n")
            return "Invalid", "Invalid"

    for _, row in openings_df.iterrows():
        opening_moves = row['uci_moves'].split()
        if moves[:len(opening_moves)] == opening_moves:
            return row['ECO'], row['name']

    return "Unknown", "Unknown"


def update_missing_openings():
    """Обновява дебютите в базата данни и проверява дали промяната е реална."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM games WHERE eco = 'Unknown' OR opening = 'Unknown'")
    games_before = cursor.fetchone()[0]
    print(f"📊 Преди старта: {games_before} партии без дебют")

    cursor.execute("SELECT id, moves_uci FROM games WHERE eco = 'Unknown' OR opening = 'Unknown'")
    games = cursor.fetchall()

    print(f"🔍 Открити {len(games)} партии за обработка.")

    start_time = time.time()
    updated_games = []
    batch_size = 1000
    optimize_every = 10000
    invalid_count = 0
    updated_count = 0

    for index, (game_id, moves_uci) in enumerate(games, start=1):
        eco, opening = get_opening_from_moves(moves_uci)

        if eco == "Invalid":
            invalid_count += 1
            continue

        updated_games.append((eco, opening, game_id))

        if len(updated_games) >= batch_size:
            cursor.executemany("UPDATE games SET eco = ?, opening = ? WHERE id = ?", updated_games)
            conn.commit()
            updated_count += len(updated_games)
            print(f"✅ Записани {updated_count} обновени партии!")
            updated_games.clear()

        if index % optimize_every == 0:
            conn.execute("PRAGMA wal_checkpoint(FULL);")
            conn.execute("VACUUM;")
            conn.commit()
            print("⚡ Оптимизация на базата данни извършена!")

    if updated_games:
        cursor.executemany("UPDATE games SET eco = ?, opening = ? WHERE id = ?", updated_games)
        conn.commit()
        updated_count += len(updated_games)
        print(f"✅ Финално записани {updated_count} партии!")

    cursor.execute("SELECT COUNT(*) FROM games WHERE eco = 'Unknown' OR opening = 'Unknown'")
    games_after = cursor.fetchone()[0]
    print(f"📊 След изпълнението: {games_after} партии без дебют (разлика: {games_before - games_after})")

    conn.close()
    elapsed_time = time.time() - start_time
    print(f"✅ Процесът завърши за {elapsed_time:.2f} секунди!")
    print(f"⚠️ Невалидни игри: {invalid_count} (записани в {invalid_games_path})")


if __name__ == "__main__":
    update_missing_openings()
