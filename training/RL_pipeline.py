import sys
import os

# Добавяме project root към Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Сега могат да идват import-ите:
import os
import time
import csv
import random
import traceback
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from models.AlphaZero import AlphaZeroModel
from training.self_play import play_episode
from training.buffer import ReplayBuffer


# Опит за по-стабилно споделяне на тензори между процеси
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

# ==========================
# Пътища и глобални настройки
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "alpha_zero_supervised.pth")
OPTIMIZER_PATH = os.path.join(BASE_DIR, "alpha_zero_supervised_opt.pth")

LOG_DIR = os.path.join(BASE_DIR, "logs")
SELFPLAY_SAVE_DIR = os.path.join(BASE_DIR, "rl", "selfplay_data")
STATS_CSV = os.path.join(LOG_DIR, "rl_stats.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SELFPLAY_SAVE_DIR, exist_ok=True)

# multiprocessing настройки – могат да се override-нат от CLI
NUM_PROCESSES = 2
GAMES_PER_PROCESS = 1

# Draw penalty (ако target_value == 0 от self-play → леко губещо)
DRAW_VALUE_PENALTY = -0.15


# ==========================
# Worker функция за self-play
# ==========================
def _worker_self_play(
    worker_id: int,
    model_state_dict,
    sims: int,
    temperature: float,
    games_per_worker: int,
    queue,
    device_str: str = "cpu",
    verbose: bool = False,
):
    """
    Един worker:
    - зарежда CPU-safe state_dict
    - качва модела на GPU (ако device_str == "cuda:X")
    - играе games_per_worker self-play игри
    - връща всички (s, pi, v) + статистики
    """
    try:
        # Определяме устройството
        device = torch.device(device_str)

        # Ако worker-ът трябва да ползва GPU
        if device_str.startswith("cuda"):
            if ":" in device_str:
                gpu_index = int(device_str.split(":")[1])
            else:
                gpu_index = 0
            torch.cuda.set_device(gpu_index)

        # Локален модел в worker-а
        model = AlphaZeroModel().to(device)
        model.load_state_dict(model_state_dict)  # dict е CPU-only → safe за pickle
        model.eval()

        total_examples = []
        game_results = []
        game_lengths = []

        for game_idx in range(games_per_worker):
            # Локален self-play буфер (малък, за текущата партия)
            sp_buffer = ReplayBuffer(max_size=5000)

            # Играем една self-play партия
            result_str = play_episode(
                model=model,
                buffer=sp_buffer,
                device=device,
                simulations=sims,
                base_temperature=temperature,
                verbose=False,
            )

            # Вземаме (s, pi, v) от буфера
            if hasattr(sp_buffer, "buffer"):
                # training/replay_buffer.py използва self.buffer = []
                examples = list(sp_buffer.buffer)
            else:
                examples = []

            total_examples.extend(examples)
            game_results.append(result_str)
            game_lengths.append(len(examples))

            if verbose:
                print(
                    f"[Worker {worker_id}] Game {game_idx + 1}/{games_per_worker} "
                    f"→ result={result_str}, positions={len(examples)}"
                )

        # Връщаме резултат обратно в main процеса
        queue.put(
            {
                "worker_id": worker_id,
                "examples": total_examples,      # list of (state, policy_vector, value)
                "game_results": game_results,    # list of result strings
                "game_lengths": game_lengths,    # list of ints
                "error": None,
            }
        )

    except Exception as e:
        queue.put(
            {
                "worker_id": worker_id,
                "examples": [],
                "game_results": [],
                "game_lengths": [],
                "error": f"{e}\n{traceback.format_exc()}",
            }
        )


# ==========================
# Основен RL Pipeline клас
# ==========================
class RLPipeline:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        optimizer_path: str = OPTIMIZER_PATH,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        buffer_capacity: int = 200_000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Модел
        self.model = AlphaZeroModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Зареждане на модела (ако има)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                self.model.load_state_dict(state["state_dict"])
            else:
                self.model.load_state_dict(state)
            self.log(f"[INFO] Loading model from {model_path}")
        else:
            self.log(f"[INFO] No existing model found at {model_path}. Starting from scratch.")

        # Зареждане на оптимизатора (ако има)
        if os.path.exists(optimizer_path):
            opt_state = torch.load(optimizer_path, map_location=self.device)
            try:
                self.optimizer.load_state_dict(opt_state)
                self.log(f"[INFO] Resumed optimizer from {optimizer_path}")
            except Exception as e:
                self.log(f"[WARN] Could not load optimizer state: {e}")
        else:
            self.log(f"[INFO] No optimizer state found at {optimizer_path}. Starting fresh optimizer.")

        self.model_path = model_path
        self.optimizer_path = optimizer_path

        # RL Replay Buffer – използваме същия клас като за self-play,
        # но с по-голям capacity
        self.replay_buffer = ReplayBuffer(max_size=buffer_capacity)

        # Лог файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOG_DIR, f"rl_training_{timestamp}.log")

        # CSV за статистики
        if not os.path.exists(STATS_CSV):
            with open(STATS_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "episode",
                        "collected_positions",
                        "num_games",
                        "avg_length",
                        "draw_rate",
                        "win_rate_white",
                        "win_rate_black",
                    ]
                )

    # ----------------- Logging helper -----------------
    def log(self, msg: str):
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line)
        try:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    # ----------------- Model saving -----------------
    def save_model(self, suffix: str = ""):
        model_path = self.model_path if not suffix else self.model_path.replace(".pth", f"_{suffix}.pth")
        opt_path = self.optimizer_path if not suffix else self.optimizer_path.replace(".pth", f"_{suffix}.pth")

        state_dict = self.model.state_dict()
        torch.save(state_dict, model_path)
        torch.save(self.optimizer.state_dict(), opt_path)
        self.log(f"[INFO] Model saved → {model_path}")

    # ----------------- Self-play collection -----------------
    def _collect_self_play_parallel(
        self,
        sims: int,
        temperature: float,
        processes: int,
        games_per_worker: int,
        verbose_workers: bool = False,
    ):
        """
        Стартира паралелен self-play.
        Връща:
          - all_examples: list of (state, policy_vector, value)
          - all_results: list of "1-0", "0-1", "1/2-1/2"
          - all_lengths: list of game lengths (брой позиции)
          - worker_errors: list of (worker_id, error_str)
        """
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        # CPU-only state_dict за безопасен pickle
        raw_sd = self.model.state_dict()
        model_state = {k: v.detach().cpu() for k, v in raw_sd.items()}

        worker_device = "cuda" if torch.cuda.is_available() else "cpu"

        procs = []
        worker_errors = []

        for wid in range(processes):
            if worker_device.startswith("cuda"):
                gpu_id = wid % torch.cuda.device_count()
                device_str = f"cuda:{gpu_id}"
            else:
                device_str = "cpu"

            p = ctx.Process(
                target=_worker_self_play,
                args=(
                    wid,
                    model_state,
                    sims,
                    temperature,
                    games_per_worker,
                    queue,
                    device_str,
                    verbose_workers,
                ),
            )
            p.start()
            procs.append(p)

        all_examples = []
        all_results = []
        all_lengths = []

        for _ in range(processes):
            data = queue.get()

            if data["error"]:
                print(f"[PAR] Worker {data['worker_id']} error:")
                print(data["error"])
                worker_errors.append((data["worker_id"], data["error"]))
                continue

            all_examples.extend(data["examples"])
            all_results.extend(data["game_results"])
            all_lengths.extend(data["game_lengths"])

            print(f"[PAR] Worker {data['worker_id']} returned {len(data['examples'])} positions.")

        # Чистим worker процесите
        for p in procs:
            p.join(timeout=1.0)
            if p.is_alive():
                print(f"[PAR] Terminating unresponsive worker PID={p.pid}")
                p.terminate()

        return all_examples, all_results, all_lengths, worker_errors

    # ----------------- RL Train step -----------------
    def train_step(self, batch_size: int = 64):
        """
        Един training step върху replay_buffer.
        Печата [TRAIN] и [DIAG] логове.
        """
        if len(self.replay_buffer) == 0:
            self.log("[TRAIN] Not enough data. Replay buffer is empty.")
            return None

        states, policies, values = self.replay_buffer.sample(batch_size)
        if states is None or len(states) == 0:
            self.log("[TRAIN] Not enough data or sample() returned empty.")
            return None

        # States → тензор [B, C, H, W]
        state_tensors = []
        for s in states:
            if isinstance(s, torch.Tensor):
                st = s.detach().cpu().float()
            else:
                st = torch.tensor(s, dtype=torch.float32)

            # ако е (H,W,C) → (C,H,W)
            if st.ndim == 3 and st.shape[0] not in (1, 3, 15) and st.shape[-1] in (1, 3, 15):
                st = st.permute(2, 0, 1)

            state_tensors.append(st)

        states_tensor = torch.stack(state_tensors, dim=0).to(self.device)

        # Policies → [B, 4672]
        policy_tensors = []
        for p in policies:
            if isinstance(p, torch.Tensor):
                pt = p.detach().cpu().float()
            else:
                arr = np.array(p, dtype=np.float32)
                pt = torch.from_numpy(arr)
            policy_tensors.append(pt)

        target_policies = torch.stack(policy_tensors, dim=0).to(self.device)

        # Values → [B]
        target_values = torch.tensor(
            [0.0 if (v is None) else float(v) for v in values],
            dtype=torch.float32,
            device=self.device,
        )

        # Draw penalty shaping: ако target_value == 0 → леко negative
        zero_mask = (target_values.abs() < 1e-6)
        target_values[zero_mask] = DRAW_VALUE_PENALTY

        # Forward
        self.model.train()
        out_policy, out_value = self.model(states_tensor)  # out_policy: [B,4672], out_value: [B,1]

        out_value = out_value.view(-1)  # [B]

        log_probs = F.log_softmax(out_policy, dim=1)
        # Policy loss (soft targets cross-entropy)
        policy_loss = -torch.mean(torch.sum(target_policies * log_probs, dim=1))

        # Value loss (MSE)
        value_loss = F.mse_loss(out_value, target_values)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Accuracy: argmax на policy
        with torch.no_grad():
            pred_idx = out_policy.argmax(dim=1)
            target_idx = target_policies.argmax(dim=1)
            correct = (pred_idx == target_idx).float()
            acc = correct.mean().item()

            vals_cpu = target_values.detach().cpu().numpy()
            neg = (vals_cpu < -0.5).sum()
            zero = ((vals_cpu >= -0.5) & (vals_cpu <= 0.5)).sum()
            pos = (vals_cpu > 0.5).sum()

        stats = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "acc": acc,
            "batch": len(states),
            "pos": int(pos),
            "zero": int(zero),
            "neg": int(neg),
        }

        return stats

    # ----------------- Главен RL цикъл -----------------
    def run_training(
        self,
        episodes: int = 10,
        train_steps: int = 20,
        batch_size: int = 64,
        sims: int = 200,
        temperature: float = 1.0,
        num_processes: int = NUM_PROCESSES,
        games_per_worker: int = GAMES_PER_PROCESS,
    ):
        self.log(f"[RL] Starting training for {episodes} episodes...")

        for ep in range(1, episodes + 1):
            self.log(f"\n[EP {ep}] Collecting self-play data...")

            start_time = time.time()
            examples, game_results, game_lengths, errors = self._collect_self_play_parallel(
                processes=num_processes,
                games_per_worker=games_per_worker,
                sims=sims,
                temperature=temperature,
                verbose_workers=False,
            )
            elapsed = time.time() - start_time

            if errors:
                self.log(f"[EP {ep}] Some workers reported errors (see above).")

            if not examples:
                self.log(f"[EP {ep}] No examples collected this round.")
            else:
                # Добавяме в RL буфера – примерите са (s, pi, v)
                for (s, p, v) in examples:
                    self.replay_buffer.push(s, p, v)

                self.log(
                    f"[EP {ep}] Collected {len(examples)} examples "
                    f"(ReplayBuffer size: {len(self.replay_buffer)}) in {elapsed:.1f}s"
                )

            # Статистика за игрите
            num_games = len(game_results)
            if num_games > 0:
                lengths = [l for l in game_lengths if l is not None]
                avg_len = float(np.mean(lengths)) if lengths else 0.0

                draws = sum(1 for r in game_results if r == "1/2-1/2")
                whites = sum(1 for r in game_results if r == "1-0")
                blacks = sum(1 for r in game_results if r == "0-1")

                draw_rate = draws / num_games
                win_white = whites / num_games
                win_black = blacks / num_games

                self.log(
                    f"[STATS] Games: {num_games} | Avg length: {avg_len:.1f} | "
                    f"Draws: {draws} ({draw_rate*100:.1f}%) | "
                    f"White wins: {whites} ({win_white*100:.1f}%) | "
                    f"Black wins: {blacks} ({win_black*100:.1f}%)"
                )

                # Запис в CSV
                with open(STATS_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            datetime.now().isoformat(),
                            ep,
                            len(examples),
                            num_games,
                            avg_len,
                            draw_rate,
                            win_white,
                            win_black,
                        ]
                    )
            else:
                self.log("[STATS] No games played this episode.")

            # -------- TRAINING --------
            for step in range(train_steps):
                stats = self.train_step(batch_size=batch_size)
                if stats is None:
                    self.log("[TRAIN] Not enough data or NaN loss.")
                    break

                self.log(
                    f"[TRAIN] Step {step+1}/{train_steps} | "
                    f"Loss={stats['loss']:.4f} | "
                    f"Policy={stats['policy_loss']:.4f} | "
                    f"Value={stats['value_loss']:.4f} | "
                    f"Acc={stats['acc']*100:.2f}%"
                )
                self.log(
                    f"[DIAG] Batch values -> pos:{stats['pos']} "
                    f"zero:{stats['zero']} neg:{stats['neg']} (batch {stats['batch']})"
                )

            # Запис на модела след епизода
            self.save_model(suffix=f"ep{ep}")

        # Финален save
        self.save_model(suffix="final")
        self.log("[RL] Training complete!")


# ==========================
# CLI launcher
# ==========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaZero RL Training Pipeline")

    parser.add_argument("--episodes", type=int, default=20, help="Number of RL episodes (self-play + training)")
    parser.add_argument("--train_steps", type=int, default=15, help="Gradient steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--sims", type=int, default=40, help="MCTS simulations per move")
    parser.add_argument("--temperature", type=float, default=1.50, help="Root sampling temperature")
    parser.add_argument("--processes", type=int, default=NUM_PROCESSES, help="Number of self-play worker processes")
    parser.add_argument(
        "--games_per_worker", type=int, default=GAMES_PER_PROCESS, help="Number of games per worker per episode"
    )

    args = parser.parse_args()

    pipeline = RLPipeline()

    print("=======================================")
    print("🚀 Starting AlphaZero RL Training...")
    print(f" Episodes     : {args.episodes}")
    print(f" Train Steps  : {args.train_steps}")
    print(f" Batch Size   : {args.batch_size}")
    print(f" MCTS Sims    : {args.sims}")
    print(f" Temperature  : {args.temperature}")
    print(f" Processes    : {args.processes}")
    print(f" Games/Worker : {args.games_per_worker}")
    print("=======================================")

    pipeline.run_training(
        episodes=args.episodes,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        sims=args.sims,
        temperature=args.temperature,
        num_processes=args.processes,
        games_per_worker=args.games_per_worker,
    )

    print("=======================================")
    print("🎉 Training Completed Successfully")
    print("=======================================")
