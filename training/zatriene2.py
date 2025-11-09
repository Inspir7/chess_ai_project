import numpy as np
from pathlib import Path

test_dir = Path("/home/presi/projects/chess_ai_project/training/test")

for prefix in ["test_labeled_states_", "test_labeled_policies_", "test_labeled_values_"]:
    files = sorted(test_dir.glob(f"{prefix}*.npy"))
    arrays = [np.load(f) for f in files]
    data = np.concatenate(arrays, axis=0)
    print(f"{prefix} shape: {data.shape}")
