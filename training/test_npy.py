import numpy as np
import os

# Промени пътя, ако е нужно
folder_path = "C:/Users/prezi/PycharmProjects/chess_ai_project/data/datasets/train"

# Взимаме първия .npy файл в папката
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):
        full_path = os.path.join(folder_path, filename)
        print(f"Проверявам файл: {filename}")

        data = np.load(full_path, allow_pickle=True)
        print(f"\nТип на целия обект: {type(data)}")
        print(f"Брой елементи: {len(data)}")
        print(f"Тип на първия елемент: {type(data[0])}")
        print(f"\nПървият елемент: {data[0]}")

        break  # само първия файл
