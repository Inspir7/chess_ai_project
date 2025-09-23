import torch
import torch.nn as nn
import torch.optim as optim

from models.AlphaZero import AlphaZeroModel  # За type hint, ако искаш

# Запазваме оптимизатора глобално, за да се инициализира веднъж
_optimizer = None


def train_on_batch(model, episode_data, device):
    global _optimizer

    if _optimizer is None:
        _optimizer = optim.Adam(model.parameters(), lr=1e-4)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    model.train()

    states, target_policies, target_values = zip(*episode_data)
    states = torch.stack(states).to(device)
    target_policies = torch.stack(target_policies).to(device)
    target_values = torch.tensor(target_values, dtype=torch.float32, device=device)

    pred_policies, pred_values = model(states)

    # За CrossEntropyLoss се очаква индексите на най-високите вероятности
    target_policy_indices = torch.argmax(target_policies, dim=1)

    loss_p = policy_loss_fn(pred_policies, target_policy_indices)
    loss_v = value_loss_fn(pred_values.squeeze(), target_values)
    loss = loss_p + loss_v

    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()

    return loss_p.item(), loss_v.item()
