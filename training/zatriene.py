import numpy as np

data = np.load("/home/presi/projects/chess_ai_project/data/datasets/train/train_tensors_0.npy", allow_pickle=True)
print(type(data))
print(data.shape)


first_example = data[0]
state, policy, value = first_example
print("State shape:", state.shape)
print("Policy shape:", policy.shape)
print("Value:", value)
