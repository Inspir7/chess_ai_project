import numpy as np
from pathlib import Path

def load_all_values(folder):
    value_arrays = []
    for file in sorted(Path(folder).glob("test_labeled_values_*.npy")):
        values = np.load(file)
        value_arrays.append(values)
    return np.concatenate(value_arrays)

def main():
    folder = "C:\\Users\\prezi\\PycharmProjects\\chess_ai_project\\training\\test"
    values = load_all_values(folder)

    print(f"\nОбщо value примери: {len(values)}")
    print(f"Минимална стойност: {np.min(values)}")
    print(f"Максимална стойност: {np.max(values)}")
    print(f"Уникални стойности: {np.unique(values, return_counts=True)}")

    print("\nПроцентно разпределение:")
    unique, counts = np.unique(values, return_counts=True)
    for val, count in zip(unique, counts):
        perc = 100 * count / len(values)
        print(f"  Value = {val:.1f} → {count} примера ({perc:.2f}%)")

if __name__ == "__main__":
    main()
