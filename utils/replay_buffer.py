# utils/replay_buffer.py
import os
import pickle

def load_replay_buffer(filepath="utils/replays/replay_buffer.pkl"):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return []
    with open(filepath, "rb") as f:
        try:
            return pickle.load(f)
        except EOFError:
            return []

def save_replay_buffer(buffer, filepath="utils/replays/replay_buffer.pkl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(buffer, f)
