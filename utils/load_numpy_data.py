import os
import numpy as np

def load_npy_from_directory(directory):
    print(f"\n🔍 Зареждане от директория: {directory}")

    # Проверка дали директорията съществува
    if not os.path.exists(directory):
        raise FileNotFoundError(f"❌ Директорията не съществува: {directory}")

    # Всички файлове в директорията
    files = os.listdir(directory)
    print(f"📂 Намерени файлове: {files}")

    if len(files) == 0:
        raise ValueError(f"⚠️ Няма файлове в директорията: {directory}")

    # Филтриране по тип
    state_files = sorted([f for f in files if "_state" in f])
    policy_files = sorted([f for f in files if "_policies" in f])
    value_files = sorted([f for f in files if "_value" in f])

    print(f"🧠 state файлове: {state_files}")
    print(f"🎯 policy файлове: {policy_files}")
    print(f"💎 value файлове: {value_files}")

    # Проверка дали трите типа файлове са намерени
    if not state_files or not policy_files or not value_files:
        raise ValueError("❗ Не са намерени всички видове .npy файлове (state, policy, value). Провери имената!")

    # Зареждане на файловете
    all_states = [np.load(os.path.join(directory, f)) for f in state_files]
    all_policies = [np.load(os.path.join(directory, f)) for f in policy_files]
    all_values = [np.load(os.path.join(directory, f)) for f in value_files]

    # Обединяване
    return (
        np.concatenate(all_states, axis=0),
        np.concatenate(all_policies, axis=0),
        np.concatenate(all_values, axis=0)
    )
