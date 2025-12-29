import sys
import os
import math  # за Elo proxy
import copy

# ==========================
# Python path → project root
# ==========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import csv
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim

from models.AlphaZero import AlphaZeroModel
from training.self_play import play_episode
from training.buffer import ReplayBuffer

# ============================================================
# Пътища и директории
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(BASE_DIR, "logs")
SELFPLAY_SAVE_DIR = os.path.join(BASE_DIR, "rl", "selfplay_data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "rl", "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SELFPLAY_SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

STATS_CSV = os.path.join(LOG_DIR, "rl_stats.csv")

# Supervised моделът, който обучавахме с train.py
SUPERVISED_INIT_PATH = os.path.join(BASE_DIR, "alpha_zero_supervised.pth")

# Основен RL модел и optimizer (тук се пазят чекпойнтовете)
RL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "alpha_zero_rl_checkpoint_final.pth")
RL_OPT_PATH = os.path.join(CHECKPOINT_DIR, "alpha_zero_rl_opt.pth")

# multiprocessing настройки (default-и – могат да се override-нат от CLI)
NUM_PROCESSES = 2
GAMES_PER_PROCESS = 1

# Draw shaping – ВРЕМЕННО изключено (иначе bias-ва value head-а)
# CHANGED: Вече не се ползва тук, защото self_play.py връща Material Score
DRAW_VALUE_PENALTY = 0.0


# ============================================================
# WORKER: SELF-PLAY генератор
# ============================================================
def _worker_self_play(
        worker_id,
        model_state_dict,
        frozen_state_dict,
        sims,
        temperature,
        games_per_worker,
        queue,
        device_str="cpu",
):
    """
    Един worker:
      • зарежда state_dict (CPU-safe)
      • инициализира модела
      • играе няколко self-play игри
      • връща всички (state, pi, value) директно (NO buffer)
    """
    try:
        device = torch.device(device_str)
        if device_str.startswith("cuda"):
            torch.cuda.set_device(int(device_str.split(":")[1]))

        # Локален модел в worker-а
        model = AlphaZeroModel().to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        # CHANGED: Зареждаме frozen модела (Supervised)
        frozen_model = AlphaZeroModel().to(device)
        frozen_model.load_state_dict(frozen_state_dict)
        frozen_model.eval()

        for p in frozen_model.parameters():
            p.requires_grad = False

        total_examples = []
        game_results = []
        game_lengths = []

        for _ in range(games_per_worker):
            # play_episode връща:
            #   examples = [(state_np, pi_np, value_float), ...]
            #   result_str = "1-0" / "0-1" / "1/2-1/2"
            #   game_len = брой полуходове

            # CHANGED: Подаваме frozen_model в play_episode
            examples, result_str, game_len = play_episode(
                model=model,
                device=device,
                frozen_model=frozen_model,  # <-- ТУК
                simulations=sims,
                base_temperature=temperature,
                verbose=False,
            )

            total_examples.extend(examples)
            game_results.append(result_str)
            game_lengths.append(game_len)

        # safe сериализация → всичко към numpy / float
        safe_examples = []
        for (s, pi, v) in total_examples:
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            if isinstance(pi, torch.Tensor):
                pi = pi.detach().cpu().numpy()
            safe_examples.append((s, pi, float(v)))

        queue.put(
            {
                "worker_id": worker_id,
                "examples": safe_examples,
                "results": game_results,
                "lengths": game_lengths,
                "error": None,
            }
        )

    except Exception as e:
        queue.put(
            {
                "worker_id": worker_id,
                "examples": [],
                "results": [],
                "lengths": [],
                "error": f"{e}\n{traceback.format_exc()}",
            }
        )


# ============================================================
# RLPipeline – главният RL контролер
# ============================================================
class RLPipeline:
    def __init__(
            self,
            supervised_path: str = SUPERVISED_INIT_PATH,
            rl_model_path: str = RL_MODEL_PATH,
            optimizer_path: str = RL_OPT_PATH,
            lr: float = 0.000035,
            weight_decay: float = 1e-4,
            buffer_capacity: int = 1_200_000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AlphaZeroModel().to(self.device)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # VREMENNO Хардкодваме стойността, за да игнорираме всичко друго
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.000035, weight_decay=weight_decay)
        # Фина настройка за Фаза 2 (Bootcamp)
        # Намаляваме от 0.000035 на 0.00002
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=weight_decay)

        # === ФАЗА VI: ANNEALING / STABILIZATION ===
        # Много нисък LR за фино полиране
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0000005, weight_decay=weight_decay)

        # LR scheduler ще се създаде в run_training,
        # защото там знаем episodes * train_steps (T_max за cosine).
        self.scheduler = None

        self.supervised_path = supervised_path
        self.model_path = rl_model_path
        self.optimizer_path = optimizer_path

        # --------------------- Инициализация ---------------------
        loaded_from = None

        # 1) Ако имаме RL checkpoint → резюмираме RL
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                print(f"[INFO] Loaded RL model → {self.model_path}")
                print("[INFO] Using FRESH optimizer (intentionally reset).")
                loaded_from = "rl"
            except Exception as e:
                print(f"[WARN] Failed to load RL checkpoint: {e}")

        # 2) Иначе, ако има supervised модел → стартираме RL от него
        if loaded_from is None and os.path.exists(self.supervised_path):
            try:
                self.model.load_state_dict(torch.load(self.supervised_path, map_location=self.device))
                print(f"[INFO] Loaded supervised init → {self.supervised_path}")
                print("[INFO] Using FRESH optimizer for RL (no old moments).")
                loaded_from = "supervised"
            except Exception as e:
                print(f"[WARN] Failed to load supervised model: {e}")

        # 3) Ако няма нищо → random init
        if loaded_from is None:
            print("[WARN] No RL checkpoint or supervised model found. Starting from RANDOM init.")

        self.replay_buffer = ReplayBuffer(max_size=buffer_capacity)

        # Лог файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOG_DIR, f"rl_training_{timestamp}.log")

        # CSV заглавие (ако липсва)
        if not os.path.exists(STATS_CSV):
            with open(STATS_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "episode",
                        "positions",
                        "games",
                        "avg_len",
                        "draws",
                        "white",
                        "black",
                        "elo_proxy",
                    ]
                )

    # ---------------------------------------------------------
    # Logging helper
    # ---------------------------------------------------------
    def log(self, msg: str):
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    # ---------------------------------------------------------
    # Safe model saving
    # ---------------------------------------------------------
    """
        • Без suffix → презаписва основния RL модел (alpha_zero_rl_main.pth)
        • С suffix (ep10, ep15, final...) → checkpoint в папката checkpoints/
    """

    def save_model(self, suffix: str = ""):
        if suffix == "":
            model_path = self.model_path
            opt_path = self.optimizer_path
        else:
            model_path = os.path.join(
                CHECKPOINT_DIR, f"alpha_zero_rl_checkpoint_{suffix}.pth"
            )
            opt_path = os.path.join(
                CHECKPOINT_DIR, f"alpha_zero_rl_opt_checkpoint_{suffix}.pth"
            )

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), opt_path)

        self.log(f"[INFO] Saved model → {model_path}")
        self.log(f"[INFO] Saved optimizer → {opt_path}")

    # ---------------------------------------------------------
    # Parallel self-play
    # ---------------------------------------------------------
    def _collect_self_play_parallel(self, sims, temperature, processes, games_per_worker):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        # CPU-safe state dict за worker-ите
        model_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        # 🔒 Frozen model = supervised init (НЕ го променяме)
        frozen_state = torch.load(
            self.supervised_path,
            map_location="cpu"
        )

        procs = []
        for wid in range(processes):
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device_str = f"cuda:{wid % torch.cuda.device_count()}"
            else:
                device_str = "cpu"

            p = ctx.Process(
                target=_worker_self_play,
                args=(wid, model_state, frozen_state, sims, temperature, games_per_worker, queue, device_str),
            )
            p.start()
            procs.append(p)

        all_examples, all_results, all_lengths = [], [], []

        for _ in range(processes):
            data = queue.get()

            if data["error"]:
                print(f"[WORKER ERROR] → {data['error']}")
                continue

            all_examples.extend(data["examples"])
            all_results.extend(data["results"])
            all_lengths.extend(data["lengths"])

        # Чисто спиране
        for p in procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

        return all_examples, all_results, all_lengths

    # ---------------------------------------------------------
    # Training step (1 mini-batch от replay buffer)
    # ---------------------------------------------------------
    def train_step(self, batch_size=64):
        if len(self.replay_buffer) == 0:
            return None

        states, policies, values = self.replay_buffer.sample(batch_size)
        if not states:
            return None

        # States → Tensor
        state_tensors = []
        for s in states:
            if isinstance(s, torch.Tensor):
                st = s.detach().clone().float()
            else:
                st = torch.tensor(s, dtype=torch.float32)

            # ако е (H, W, C) → (C, H, W)
            if st.ndim == 3 and st.shape[-1] in (1, 3, 15):
                st = st.permute(2, 0, 1)

            state_tensors.append(st)

        states_tensor = torch.stack(state_tensors).to(self.device)

        # Policies (π vector 4672)
        target_policies = torch.stack(
            [
                p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)
                for p in policies
            ]
        ).to(self.device)

        # Values
        target_values = torch.tensor(
            [float(v) if v is not None else 0.0 for v in values],
            device=self.device,
            dtype=torch.float32,
        )

        # CHANGED: Премахнато хардкоднатото Draw Penalty, защото self_play.py
        # вече връща material-based score (не нули) при реми.
        # for i, v in enumerate(target_values):
        #     if abs(v.item()) < 1e-6:
        #         target_values[i] = -0.05

        self.model.train()
        out_policy, out_value = self.model(states_tensor)
        out_value = out_value.view(-1)

        log_probs = F.log_softmax(out_policy, dim=1)
        policy_loss = -torch.mean(torch.sum(target_policies * log_probs, dim=1))
        value_loss = torch.mean((out_value - target_values) ** 2)
        # loss = policy_loss + value_loss - default
        loss = 1.2 * policy_loss + 0.8 * value_loss  # mid state

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        # LR scheduler (Cosine) – стъпка на всяка mini-batch
        if self.scheduler is not None:
            self.scheduler.step()

        # Текущ LR за логване
        current_lr = self.optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            pred = out_policy.argmax(dim=1)
            targ = target_policies.argmax(dim=1)
            acc = (pred == targ).float().mean().item()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "acc": acc,
            "lr": current_lr,
        }

    # ---------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------
    def run_training(
            self,
            episodes=10,
            train_steps=20,
            batch_size=64,
            sims=200,
            temperature=1.0,
            num_processes=NUM_PROCESSES,
            games_per_worker=GAMES_PER_PROCESS,
    ):
        """
        temperature: използва се като НАЧАЛНА температура.
        Ще направим линейно намаляване към по-ниска температура в края на training-а.

        sims: брой MCTS симулации в началото; по време на training-а
        ще ги увеличаваме автоматично (automatic MCTS sims schedule).
        """
        self.log(f"[RL] Starting training for {episodes} episodes...")

        # ------------------------
        # Cosine LR scheduler
        # ------------------------
        total_train_steps = max(1, episodes * train_steps)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_train_steps,
            eta_min=1e-6,
        )

        # Temperature schedule: линейно от temperature → final_temp
        temp_start = float(temperature)
        # temp_final = max(0.3, temp_start * 0.25)  # напр. 1.5 → 0.375
        temp_final = max(0.90, temp_start * 0.75)  # започва от 1.3 → пада към ~1.0

        # Automatic MCTS sims schedule: от sims_start → sims_end
        sims_start = int(sims)
        # sims_end = max(int(sims * 4), sims_start + 300)  # напр. 40 → поне 300+
        # расте по-малко и никога не става огромно
        sims_end = min(sims_start + 320, 400)

        try:
            global_step = 0

            for ep in range(1, episodes + 1):
                # ---------------- ПРОГРЕС (0 → 1) ----------------
                if episodes > 1:
                    progress = (ep - 1) / (episodes - 1)
                else:
                    progress = 1.0

                # Температура за този епизод
                current_temp = temperature
                # = temp_start + (temp_final - temp_start) * progress -> izkluchvane na scheduler

                # Никога да не пада под разумен минимум (предпазва от реми-колапс)
                # current_temp = max(current_temp, 1.05) - early state
                #current_temp = max(current_temp, 0.8)  # mid-state -> deleted for last phase

                # Sims за този епизод (нарастват с епизода)
                #current_sims = int(sims_start + (sims_end - sims_start) * progress) -> deleted for annealing phase
                current_sims = sims

                self.log(
                    f"\n[EP {ep}] Collecting self-play... "
                    f"(temperature={current_temp:.2f}, sims={current_sims})"
                )

                # ---------------- SELF-PLAY ----------------
                start = time.time()
                examples, results, lengths = self._collect_self_play_parallel(
                    sims=current_sims,
                    temperature=current_temp,
                    processes=num_processes,
                    games_per_worker=games_per_worker,
                )
                elapsed = time.time() - start

                # Save raw self-play на всеки 5 епизода
                if ep % 5 == 0:
                    ep_path = os.path.join(SELFPLAY_SAVE_DIR, f"episode_{ep}.pt")
                    torch.save({"examples": examples, "results": results, "lengths": lengths}, ep_path)
                    self.log(f"[EP {ep}] Saved episode → {ep_path}")

                # Add към replay buffer
                for (s, p, v) in examples:
                    self.replay_buffer.push(s, p, v)

                # Статистика за игрите + Elo proxy
                if results:
                    draws = results.count("1/2-1/2")
                    # Старо (грешно за новите надписи):
                    # whites = results.count("1-0")
                    # blacks = results.count("0-1")

                    # НОВО (работи и с "1-0" и с "1-0 (Adj)"):
                    whites = sum(1 for r in results if r.startswith("1-0"))
                    blacks = sum(1 for r in results if r.startswith("0-1"))

                    avg_len = float(np.mean(lengths))
                    total_games = len(results)

                    # Elo proxy: white score vs 0.5 baseline
                    white_score = whites + 0.5 * draws
                    score_frac = white_score / max(1, total_games)

                    # Ако НЯМА решени партии (само ремита) → Elo не е дефиниран
                    if whites == 0 and blacks == 0:
                        elo_proxy = float("nan")
                    else:
                        eps = 1e-3
                        score_clipped = min(max(score_frac, eps), 1.0 - eps)
                        elo_proxy = 400.0 * math.log10(score_clipped / (1.0 - score_clipped))

                    # Стринг за лог
                    if math.isnan(elo_proxy):
                        elo_str = "N/A"
                    else:
                        elo_str = f"{elo_proxy:.1f}"

                    self.log(
                        f"[STATS] Games={total_games} | AvgLen={avg_len:.1f} "
                        f"| Draws={draws} | W={whites} | B={blacks} "
                        f"| EloProxy={elo_str} | CollectTime={elapsed:.1f}s"
                    )

                    # Пишем в CSV (nan остава nan за pandas)
                    with open(STATS_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                datetime.now().isoformat(),
                                ep,
                                len(examples),
                                total_games,
                                avg_len,
                                draws,
                                whites,
                                blacks,
                                elo_proxy,
                            ]
                        )

                # ---------------- TRAIN STEPS ----------------
                for step in range(train_steps):
                    stats = self.train_step(batch_size=batch_size)
                    global_step += 1

                    if stats:
                        self.log(
                            f"[TRAIN] EP{ep} Step{step + 1}/{train_steps} | "
                            f"Loss={stats['loss']:.4f} | "
                            f"Policy={stats['policy_loss']:.4f} | "
                            f"Value={stats['value_loss']:.4f} | "
                            f"Acc={stats['acc'] * 100:.2f}% | "
                            f"LR={stats['lr']:.6f}"
                        )

                # Save основен RL модел (overwrite)
                self.save_model()

                # Checkpoint на всеки 5 епизода
                if ep % 5 == 0:
                    self.save_model(suffix=f"ep{ep}")

            # Финален модел
            self.save_model(suffix="final")
            self.log("[RL] Training complete!")

        except KeyboardInterrupt:
            self.log("[WARN] Training INTERRUPTED!")
            self.save_model(suffix="interrupted")
            self.log("[WARN] Interrupted checkpoint saved.")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaZero RL Training Pipeline")

    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--train_steps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.20,
        help="Начална температура за self-play; ще се намаля линейно към по-ниска.",
    )
    parser.add_argument("--processes", type=int, default=NUM_PROCESSES)
    parser.add_argument("--games_per_worker", type=int, default=GAMES_PER_PROCESS)

    args = parser.parse_args()

    pipeline = RLPipeline()

    print("=======================================")
    print("🚀 Starting AlphaZero RL Training...")
    print(f" Episodes     : {args.episodes}")
    print(f" Train Steps  : {args.train_steps}")
    print(f" Batch Size   : {args.batch_size}")
    print(f" MCTS Sims    : {args.sims} (start; will auto-increase mildly)")
    print(f" Temp Start   : {args.temperature}")
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