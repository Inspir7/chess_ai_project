# В някой тестов файл или в RL pipeline:
from training.RL_pipeline import model, device
from training.self_play import play_episode
examples = play_episode(model, device, simulations=50, temperature=1.0)
