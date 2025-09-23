import os
import multiprocessing


def count_games_in_file_fast(file_path):
    """Оптимизирано броене на партии чрез обработка ред по ред (за големи PGN файлове)."""
    try:
        count = 0
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                if line.startswith("[Event "):  # Всяка партия в PGN започва с този таг
                    count += 1
                    if count % 100000 == 0:  # Принтиране на прогрес на всеки 100K партии
                        print(f"📍 {file_path}: {count} партии обработени...")

        print(f"✅ Обработен файл: {file_path} | Намерени партии: {count}")
        return count
    except Exception as e:
        print(f"❌ Грешка при обработката на {file_path}: {e}")
        return 0


def count_games_in_folder(folder_path):
    """Използва multiprocessing за паралелно броене на партиите в PGN файловете с логове."""
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pgn")]

    print(f"📂 Намерени PGN файлове: {len(files)}")

    with multiprocessing.Pool(processes=6) as pool:  # Ограничаваме процесите до 6
        results = []
        for i, count in enumerate(pool.imap_unordered(count_games_in_file_fast, files), 1):
            results.append(count)
            if i % 5 == 0:  # Принтиране на прогреса на всеки 5 обработени файла
                print(f"📊 Обработени файлове: {i}/{len(files)}")

    total_games = sum(results)
    print(f"🎯 Общо намерени партии: {total_games}")
    return total_games


if __name__ == "__main__":
    folder_path = "data/pgn"  # Замени с реалния път до PGN файловете
    total_games = count_games_in_folder(folder_path)
    print(f"🏁 Край! Общ брой изиграни партии: {total_games}")
