import os
import multiprocessing

def list_pgn_files(folder_path):
    """Функция за изброяване на PGN файлове в дадена папка."""
    try:
        if not os.path.exists(folder_path):
            print(f"⚠️ Папката '{folder_path}' не съществува!")
            return

        pgn_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pgn")]

        if not pgn_files:
            print("📂 Няма PGN файлове в тази папка.")
        else:
            print("📂 Намерени PGN файлове:")
            for file in pgn_files:
                print(f" - {file}")

    except Exception as e:
        print(f"❌ Грешка: {e}")

if __name__ == "__main__":
    folder_path = r"D:\pgn2"  # Въвеждаш правилния път

    # Стартираме паралелен процес
    process = multiprocessing.Process(target=list_pgn_files, args=(folder_path,))
    process.start()

    print("🚀 Процесът за изброяване на файловете стартира във фонов режим!")

    # Ако искаш да изчакаш да завърши:
    # process.join()
