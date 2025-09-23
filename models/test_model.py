from models.AlphaZero import AlphaZeroModel
import torch

# Създай модела
model = AlphaZeroModel()

# Създай примерен вход: 1 позиция, 15 канала, 8x8
sample_input = torch.randn(1, 15, 8, 8)

# Предсказания
policy_output, value_output = model(sample_input)

# Отпечатай формата на изходите
print("Policy shape:", policy_output.shape)  # Очаквано: [1, 64]
print("Value:", value_output.item())         # Очаквано: стойност между -1 и 1
