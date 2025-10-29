import os
import numpy as np

# Път към твоите .npy файлове
base_dir = r"/home/presi/projects/chess_ai_project/training"

splits = ["train", "validation", "test"]

for split in splits:
    split_path = os.path.join(base_dir, split)
    if not os.path.exists(split_path):
        print(f"⚠️ Папката {split} не съществува.")
        continue

    npy_files = [f for f in os.listdir(split_path) if f.endswith(".npy")]
    print(f"\n📁 {split.upper()} ({len(npy_files)} файла)")

    if not npy_files:
        continue

    # Зареждаме само първия файл за пример
    path = os.path.join(split_path, npy_files[0])
    data = np.load(path, allow_pickle=True)
    print(f"➡️ Зареден файл: {npy_files[0]}")
    print(f"Тип данни: {type(data)}")

    # Проверка какво съдържа
    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print("Примерен елемент:")
        print(data[0])

    elif isinstance(data, dict):
        print(f"Ключове: {list(data.keys())}")
        for k, v in data.items():
            print(f"  {k}: shape={np.array(v).shape}")

    elif isinstance(data, list) and isinstance(data[0], dict):
        print(f"Списък от {len(data)} елемента. Примерен ключов набор:")
        print(data[0].keys())

    else:
        print("❓ Непозната структура, трябва да видим конкретния пример.")
