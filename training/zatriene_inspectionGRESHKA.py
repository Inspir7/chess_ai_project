import torch
from training.RL_pipeline import RLPipeline

# --- Устройство ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Device:", device)

# --- Зареждане на RL pipeline ---
pipeline = RLPipeline(device=device)
model = pipeline.model
buffer = pipeline.buffer
model.eval()

print("✅ Моделът е успешно зареден от RL Pipeline!")

# --- Генериране на self-play ---
print("\n🎮 Генериране на self-play епизод...")
examples = pipeline.self_play_episode(mcts_sims=30)
print(f"✅ Генерирани {len(examples)} позиции от self-play")

# --- Добавяне в буфера ---
pipeline.add_examples(examples)
print(f"📦 Buffer size: {len(buffer)}")

# --- Вземаме батч за forward pass ---
batch_size = min(8, len(buffer))
states, policies, values = buffer.sample(batch_size)
states, policies, values = states.to(device), policies.to(device), values.to(device)

# --- Forward pass ---
with torch.no_grad():
    pred_policy, pred_value = model(states)

# --- Диагностика на загубите и outputs ---
policy_loss = torch.nn.functional.cross_entropy(pred_policy, torch.argmax(policies, dim=1))
value_loss = torch.nn.functional.mse_loss(pred_value, values)

print("\n📊 Forward pass diagnostics:")
print(f"Policy loss: {policy_loss.item():.4f}")
print(f"Value loss: {value_loss.item():.4f}")
print(f"Policy output min/max: {pred_policy.min().item():.4f}/{pred_policy.max().item():.4f}")
print(f"Value output min/max: {pred_value.min().item():.4f}/{pred_value.max().item():.4f}")

# --- Опционално: печат на примерен ход и вероятност ---
example_pi = examples[0][1]
action = torch.argmax(example_pi).item()
print(f"\n🎯 Примерен move index: {action}, max prob: {example_pi.max().item():.4f}")
