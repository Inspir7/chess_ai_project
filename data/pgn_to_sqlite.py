import os
import sqlite3
import glob
import chess.pgn
import chess
from multiprocessing import Pool, cpu_count


def create_database(db_path):
    """Създава SQLite база данни и таблицата, ако още не съществува."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # Активиране на WAL режим за по-добра паралелна обработка
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT,
            site TEXT,
            date TEXT,
            round TEXT,
            white TEXT,
            black TEXT,
            result TEXT,
            eco TEXT,
            opening TEXT,
            fen TEXT UNIQUE,
            moves_san TEXT,
            moves_uci TEXT UNIQUE
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fen ON games (fen);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_moves_uci ON games (moves_uci);')
    conn.commit()
    conn.close()
    print("✅ Базата данни е създадена!")


def game_exists(cursor, fen, moves_uci):
    """Проверява дали партията вече съществува в базата."""
    cursor.execute("SELECT 1 FROM games WHERE fen = ? OR moves_uci = ?", (fen, moves_uci))
    return cursor.fetchone() is not None


def insert_game(game_data, db_path):
    """Добавя партия в базата, ако не съществува вече."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    fen, moves_uci = game_data[9], game_data[11]

    if game_exists(cursor, fen, moves_uci):
        print(f"⚠️ Пропусната дублираща партия ({fen[:30]}...)")
        return

    cursor.execute('''
        INSERT INTO games (event, site, date, round, white, black, result, eco, opening, fen, moves_san, moves_uci)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', game_data)
    conn.commit()
    conn.close()
    print(f"✅ Записана партия ({game_data[4]} vs {game_data[5]})")


def process_pgn_file(file_path, db_path):
    """Чете PGN файл и записва всяка партия в базата."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    processed_games = 0

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            moves_san, moves_uci = [], []
            for move in game.mainline_moves():
                moves_san.append(board.san(move))
                moves_uci.append(move.uci())
                board.push(move)

            fen = board.fen() if board else "UCI_" + " ".join(moves_uci)
            eco, opening = game.headers.get("ECO", "Unknown"), game.headers.get("Opening", "Unknown")

            game_data = (
                game.headers.get("Event", "Unknown"),
                game.headers.get("Site", "Unknown"),
                game.headers.get("Date", "Unknown"),
                game.headers.get("Round", "Unknown"),
                game.headers.get("White", "Unknown"),
                game.headers.get("Black", "Unknown"),
                game.headers.get("Result", "Unknown"),
                eco,
                opening,
                fen,
                " ".join(moves_san),
                " ".join(moves_uci)
            )

            insert_game(game_data, db_path)
            processed_games += 1

    conn.close()
    print(f"✅ Файл {file_path} обработен ({processed_games} партии)")


def remove_duplicates(db_path):
    """Открива и премахва дублирани партии от базата."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM games
        WHERE id NOT IN (
            SELECT MIN(id) FROM games GROUP BY fen, moves_uci
        )
    """)
    conn.commit()
    conn.close()
    print("🗑️ Дублиращите се партии бяха премахнати!")


def import_pgn_folder(folder_path, db_path, num_processes=None):
    """Обхожда всички PGN файлове и обработва партиите паралелно."""
    create_database(db_path)
    pgn_files = glob.glob(os.path.join(folder_path, "*.pgn"))
    print(f"📂 Намерени PGN файлове: {len(pgn_files)}")

    num_processes = num_processes or max(1, cpu_count() - 1)
    with Pool(num_processes) as pool:
        pool.starmap(process_pgn_file, [(f, db_path) for f in pgn_files])

    print("✅ Импортирането е завършено!")
    print("🔍 Проверка за дублирани партии...")
    remove_duplicates(db_path)


if __name__ == "__main__":
    folder_path = r"D:\\pgn2"
    db_path = r"C:\\Users\\prezi\\PycharmProjects\\chess_ai_project\\data\\datachess_games.db"
    import_pgn_folder(folder_path, db_path)
