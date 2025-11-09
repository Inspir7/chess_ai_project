# training/RL_pipeline.py (patched v2)
import os
import random
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime
import numpy as np
import multiprocessing as mp
import signal
import traceback
import warnings
import tempfile

# --- improve multiprocessing/tensor sharing stability ---
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    # older PyTorch or platforms may not support changing strategy; ignore
    pass

# моите модели/код (твои файлове)
from models.AlphaZero import AlphaZeroModel
from models.move_encoding import index_to_move, move_to_index
from training.mcts import MCTS
from utils.chess_utils import initial_board, board_to_tensor, game_over, result_from_perspective

# Твоите self-play и buffer (използваме ги директно)
from training.self_play import play_episode
from training.buffer import ReplayBuffer as UserReplayBuffer  # това е буферът от training/buffer.py

# ensure headless pygame in workers (helps when pygame is imported elsewhere)
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ---------- Multiprocessing configuration ----------
NUM_PROCESSES = 4
GAMES_PER_PROCESS = 1  # each worker plays 1 game per collection round
SELFPLAY_SAVE_DIR = "training/rl/selfplay_data"
LOG_DIR = "training/logs"
STATS_CSV = os.path.join(LOG_DIR, "rl_stats.csv")
WORKER_TIMEOUT_SEC = 3600  # per-worker watchdog (seconds)

# silence noisy warnings in workers (keeps logs readable)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# On some platforms we prefer spawn for multiprocessing safety with PyTorch
# We'll only call set_start_method if it hasn't been set yet
try:
    mp.get_start_method()
except RuntimeError:
    try:
        mp.set_start_method("spawn")
    except Exception:
        # already set or platform doesn't allow
        pass

# -----------------------
# Local RL replay buffer (keeps examples across episodes)
# -----------------------
class ReplayBuffer:
    """
    This is the internal replay buffer used by the RL pipeline (keeps many episodes).
    It stores tuples (state_tensor (C,H,W or H,W,C), policy_vector, value_scalar).
    We only require: push(iterable) or extend, sample(batch_size) and __len__.
    """
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, examples):
        # examples: iterable of (state, policy_vector, value)
        # if called with single triple, extend expects an iterable; try to detect both cases
        try:
            # if examples is a list/iterable of tuples -> extend
            first = next(iter(examples))
            # if we get here, it's an iterable
            self.buffer.extend(examples)
        except TypeError:
            # single tuple passed
            self.buffer.append(examples)
        except StopIteration:
            # empty iterable
            pass

    def sample(self, batch_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(self.buffer) == 0:
            raise ValueError("buffer empty")
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)

        # Convert states -> tensors, ensure channel-first (C,H,W)
        states_t = []
        for s in states:
            if isinstance(s, torch.Tensor):
                st = s.detach().cpu().float()
            else:
                st = torch.tensor(s, dtype=torch.float32)
            # If HWC -> CHW (some of your utilities return (15,8,8) or (8,8,15))
            if st.ndim == 3 and st.shape[0] not in (1, 3, 15) and st.shape[-1] in (1, 3, 15):
                st = st.permute(2, 0, 1)
            if st.ndim == 3 and st.shape[0] not in (1, 3, 15) and st.shape[-1] == 15:
                st = st.permute(2, 0, 1)
            states_t.append(st)

        # Convert policies -> tensors (assume full-length vectors)
        policies_t = []
        for p in policies:
            if isinstance(p, torch.Tensor):
                policies_t.append(p.detach().cpu().float())
            else:
                policies_t.append(torch.tensor(p, dtype=torch.float32))

        # Values: replace None with 0.0
        values_t = torch.tensor([0.0 if (v is None) else float(v) for v in values], dtype=torch.float32).unsqueeze(1)

        states_stack = torch.stack(states_t)
        policies_stack = torch.stack(policies_t)

        return states_stack, policies_stack, values_t

    def __len__(self):
        return len(self.buffer)

# -----------------------
# Module-level worker function (picklable)
# -----------------------
def _worker_self_play(worker_id, model_state_dict, move_vector_size, sims, temperature, games_per_worker, queue, verbose=False):
    """
    Each worker:
     - creates a local AlphaZeroModel
     - loads the given state_dict onto CPU
     - creates a user ReplayBuffer (training.buffer.ReplayBuffer)
     - calls play_episode(model, buffer, device=cpu, simulations=sims, temperature=temperature)
       GAMES_PER_WORKER times
     - extracts examples from buffer.buffer and returns them (plus metadata)
    This function is designed to be picklable and robust: it posts a "started" message early
    and posts "error" with traceback if anything fails.
    """
    # minimal local imports to ensure picklability
    try:
        import torch as _torch
        import traceback as _traceback
    except Exception:
        _torch = torch
        _traceback = traceback

    has_watchdog = False
    try:
        # watchdog (POSIX). If platform doesn't support SIGALRM (Windows), skip.
        try:
            def _alarm_handler(signum, frame):
                raise TimeoutError(f"Worker {worker_id} watchdog timeout after {WORKER_TIMEOUT_SEC}s")
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(WORKER_TIMEOUT_SEC)
            has_watchdog = True
        except Exception:
            has_watchdog = False

        # Early alive signal so parent knows this worker started
        try:
            queue.put({"worker_id": worker_id, "status": "started"})
        except Exception:
            pass

        # local model (CPU)
        model = AlphaZeroModel()
        # Load safe: try to convert state tensors to cpu
        try:
            cpu_state = {k: v.cpu() for k, v in model_state_dict.items()}
            model.load_state_dict(cpu_state)
        except Exception:
            try:
                model.load_state_dict(model_state_dict)
            except Exception as e:
                tb = _traceback.format_exc()
                try:
                    queue.put({"worker_id": worker_id, "error": f"Failed to load state_dict in worker: {e}\n{tb}"})
                except Exception:
                    print(f"[Worker {worker_id}] Failed to report load error: {e}\n{tb}")
                return

        model.to(_torch.device("cpu"))
        model.eval()

        collected_examples = []
        game_results = []
        game_lengths = []

        # we will use the user's ReplayBuffer (which exposes .buffer list)
        for g in range(games_per_worker):
            user_buffer = UserReplayBuffer()  # training/buffer.ReplayBuffer()
            try:
                # run self-play; play_episode may internally do a small local training,
                # but importantly it appends (state,pi,value) to user_buffer.buffer
                result_str = play_episode(model=model, buffer=user_buffer, device=_torch.device("cpu"),
                                          simulations=sims, temperature=temperature, verbose=verbose)
            except Exception as e:
                tb = _traceback.format_exc()
                try:
                    queue.put({"worker_id": worker_id, "error": f"play_episode crashed in worker: {e}\n{tb}"})
                except Exception:
                    print(f"[Worker {worker_id}] Failed to report play_episode error: {e}\n{tb}")
                return

            # collect examples from user_buffer.buffer
            # user_buffer.buffer likely contains tuples (state_tensor, pi_vector, value)
            for (s, p, v) in list(user_buffer.buffer):
                if isinstance(s, np.ndarray):
                    s_t = _torch.tensor(s, dtype=_torch.float32)
                else:
                    s_t = s if isinstance(s, _torch.Tensor) else _torch.tensor(s, dtype=_torch.float32)
                p_t = p if isinstance(p, _torch.Tensor) else _torch.tensor(p, dtype=_torch.float32)
                s_t = s_t.detach().cpu()
                p_t = p_t.detach().cpu()

                # --- ново: коригиране на стойността спрямо perspective ---
                try:
                    from utils.chess_utils import result_from_perspective
                    v_f = result_from_perspective(v)
                except Exception:
                    v_f = 0.0 if v is None else float(v)

                collected_examples.append((s_t, p_t, v_f))

            # derive simple metadata: game result & length (if available)
            gr = 0
            if result_str == "1-0":
                gr = 1
            elif result_str == "0-1":
                gr = -1
            else:
                gr = 0
            game_results.append(gr)
            game_lengths.append(len(user_buffer.buffer))

        # Cancel alarm before returning
        try:
            if has_watchdog:
                signal.alarm(0)
        except Exception:
            pass

        # final "done" message with examples
        try:
            # --- write examples to a temporary file and send path over queue ---
            try:
                tmpf = tempfile.NamedTemporaryFile(prefix=f"worker_{worker_id}_", suffix=".pt", delete=False)
                tmp_path = tmpf.name
                tmpf.close()
            except Exception:
                tmp_path = os.path.join(tempfile.gettempdir(), f"worker_{worker_id}_examples.pt")

            try:
                # torch.save of a list of small tensors is fine; saved file will be read by master
                _torch.save({"examples": collected_examples}, tmp_path)
            except Exception:
                # fallback: try to convert tensors to cpu numpy arrays (more robust for pickle)
                serializable = []
                for (s, p, v) in collected_examples:
                    s_numpy = s.detach().cpu().numpy() if isinstance(s, _torch.Tensor) else np.array(s)
                    p_numpy = p.detach().cpu().numpy() if isinstance(p, _torch.Tensor) else np.array(p)
                    serializable.append((s_numpy, p_numpy, v))
                # save the numpy-serializable form
                _torch.save({"examples": serializable}, tmp_path)

            queue.put({
                "worker_id": worker_id,
                "examples_file": tmp_path,
                "game_results": game_results,
                "game_lengths": game_lengths,
                "count": len(collected_examples),
                "status": "done"
            })
        except Exception as e:
            tb = _traceback.format_exc()
            print(f"[Worker {worker_id}] Failed to put final result to queue: {e}\n{tb}")
            # try an alternate simple report
            try:
                queue.put({"worker_id": worker_id, "error": f"final queue.put failed: {e}\n{tb}"})
            except Exception:
                print(f"[Worker {worker_id}] Also failed to report final failure.")

    except Exception as e:
        tb = traceback.format_exc()
        try:
            queue.put({"worker_id": worker_id, "error": f"{str(e)}\n{tb}"})
        except Exception:
            print(f"[Worker {worker_id}] Fatal error and failed to report to queue: {e}\n{tb}")
    finally:
        try:
            if has_watchdog:
                signal.alarm(0)
        except Exception:
            pass

# -----------------------
# RL pipeline
# -----------------------
class RLPipeline:
    def __init__(self,
                 model_path="/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth",
                 device=None,
                 lr=1e-3,
                 weight_decay=1e-4,
                 buffer_capacity=200_000,
                 move_vector_size=4672,
                 log_dir=LOG_DIR):
        # device selection: allow string or torch.device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.model = AlphaZeroModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # keep CrossEntropyLoss object for compatibility (not used for distributional)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.model_path = model_path
        self.move_vector_size = move_vector_size

        # logging setup
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"rl_training_{now}.log")

        # ensure selfplay save dir exists
        os.makedirs(SELFPLAY_SAVE_DIR, exist_ok=True)

        # ensure stats csv exists (with header)
        if not os.path.exists(STATS_CSV):
            try:
                with open(STATS_CSV, "w", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["timestamp", "episode", "total_positions", "games", "avg_game_length", "white_rate", "black_rate", "draw_rate", "mean_loss"])
            except Exception:
                pass

        # Safe load of model
        if os.path.exists(self.model_path):
            print(f"[INFO] Loading existing model from {self.model_path}")
            try:
                state = torch.load(self.model_path, map_location=self.device)
                try:
                    self.model.load_state_dict(state)
                except Exception:
                    # fallback: be permissive if architecture changed slightly
                    self.model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[WARN] Failed to load model state_dict cleanly: {e}")
        else:
            print("[INFO] Starting with new model.")

        # Try resume optimizer state if there is one
        try:
            opt_path = self.model_path.replace(".pth", "_opt.pth")
            if os.path.exists(opt_path):
                try:
                    self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
                    print(f"[INFO] Resumed optimizer state from {opt_path}")
                except Exception:
                    print(f"[WARN] Could not load optimizer state from {opt_path}")
        except Exception:
            pass

    # defensive wrapper for move_to_index (some implementations expect (move) others (move, board))
    def _move_to_index(self, move, board):
        try:
            idx = move_to_index(move)
            return idx
        except TypeError:
            try:
                idx = move_to_index(move, board)
                return idx
            except Exception:
                return None
        except Exception:
            return None

    # defensive wrapper for index_to_move (some variants may need board)
    def _index_to_move(self, index, board):
        try:
            mv = index_to_move(index)
            if mv is None:
                # try variant with board
                try:
                    mv = index_to_move(index, board)
                except Exception:
                    mv = None
            return mv
        except TypeError:
            try:
                mv = index_to_move(index, board)
                return mv
            except Exception:
                return None
        except Exception:
            return None

    def self_play_episode(self, mcts_sims=100, temperature=1.0, verbose=False):
        """
        Single-process self-play (kept for debug/fallback).
        Returns examples list of (state_tensor [C,H,W], pi_tensor [move_dim], z scalar)
        and move_logs for debugging.
        """
        user_buffer = UserReplayBuffer()
        try:
            result_str = play_episode(model=self.model, buffer=user_buffer, device=self.device,
                                      simulations=mcts_sims, temperature=temperature, verbose=verbose)
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"[SELF-PLAY-ERR] play_episode failed: {e}\n{tb}")
            return [], []

        examples = []
        for (s, p, v) in list(user_buffer.buffer):
            # ensure tensors
            s_t = s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
            p_t = p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)
            v_f = 0.0 if v is None else float(v)
            examples.append((s_t, p_t, v_f))

        logs = {"result": result_str, "count": len(examples)}
        return examples, logs

    def add_examples(self, examples):
        self.buffer.push(examples)

    def _collect_self_play_parallel(self, num_processes=NUM_PROCESSES, games_per_worker=GAMES_PER_PROCESS, sims=50, temperature=1.0, timeout=WORKER_TIMEOUT_SEC):
        """
        Launch workers which call play_episode() and return user buffer contents.
        """
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = []
        # send state_dict (CPU) to workers
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # paths for simple diagnostics (in case of debug saving)
        started_workers = 0
        initial_results = []

        for i in range(num_processes):
            p = ctx.Process(target=_worker_self_play,
                            args=(i, model_state_dict, self.move_vector_size, sims, temperature, games_per_worker, queue, False))
            p.start()
            processes.append(p)

        combined = []
        all_game_results = []
        all_game_lengths = []
        worker_errors = []
        start_time = time.time()

        # Wait shortly for workers to announce "started" (diagnostic)
        start_deadline = time.time() + min(10.0, timeout)
        while time.time() < start_deadline and started_workers < num_processes:
            try:
                msg = queue.get(timeout=0.5)
                if isinstance(msg, dict) and msg.get("status") == "started":
                    started_workers += 1
                    self.log(f"[PAR] Worker {msg.get('worker_id')} started.")
                else:
                    # accumulate for later processing
                    initial_results.append(msg)
            except Exception:
                # nothing ready yet
                pass

        # process any initial_results we got while waiting for starts
        for msg in initial_results:
            if not isinstance(msg, dict):
                continue
            if "error" in msg:
                worker_errors.append(msg)
                self.log(f"[PAR] Worker {msg.get('worker_id')} error (early): {msg.get('error')}")
            elif msg.get("status") == "done":
                # support both legacy 'examples' and new 'examples_file'
                if msg.get("examples_file"):
                    try:
                        data = torch.load(msg.get("examples_file"))
                        exs = data.get("examples", [])
                        # if saved as numpy arrays, convert to tensors
                        safe_exs = []
                        for (s, p, v) in exs:
                            s_t = torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s.detach().cpu()
                            p_t = torch.tensor(p, dtype=torch.float32) if not isinstance(p, torch.Tensor) else p.detach().cpu()
                            v_f = 0.0 if v is None else float(v)
                            safe_exs.append((s_t, p_t, v_f))
                        combined.extend(safe_exs)
                        # try to cleanup file
                        try:
                            os.remove(msg.get("examples_file"))
                        except Exception:
                            pass
                    except Exception as e:
                        worker_errors.append({"worker_id": msg.get("worker_id"), "error": f"failed to load examples_file early: {e}"})
                else:
                    exs = msg.get("examples", [])
                    combined.extend(exs)
                try:
                    count_len = msg.get("count", len(exs) if 'exs' in locals() else 0)
                except Exception:
                    count_len = 0
                self.log(f"[PAR] (early) Worker {msg.get('worker_id')} returned {count_len} positions.")
                all_game_results.extend(msg.get("game_results", []))
                all_game_lengths.extend(msg.get("game_lengths", []))

        # Now collect up to num_processes results, with timeout
        for _ in range(num_processes):
            try:
                res = queue.get(timeout=timeout)
            except Exception:
                res = {"worker_id": None, "error": "timeout waiting for worker"}
            if "error" in res:
                worker_errors.append(res)
                self.log(f"[PAR] Worker {res.get('worker_id')} error: {res.get('error')}")
            else:
                # support both legacy 'examples' and new 'examples_file'
                examples = []
                if res.get("examples_file"):
                    fpath = res.get("examples_file")
                    try:
                        data = torch.load(fpath)
                        exs = data.get("examples", [])
                        for (s, p, v) in exs:
                            # convert numpy back to tensor if needed
                            if isinstance(s, np.ndarray):
                                s_cpu = torch.tensor(s, dtype=torch.float32)
                            elif isinstance(s, torch.Tensor):
                                s_cpu = s.detach().cpu()
                            else:
                                s_cpu = torch.tensor(s, dtype=torch.float32)
                            if isinstance(p, np.ndarray):
                                p_cpu = torch.tensor(p, dtype=torch.float32)
                            elif isinstance(p, torch.Tensor):
                                p_cpu = p.detach().cpu()
                            else:
                                p_cpu = torch.tensor(p, dtype=torch.float32)
                            v_f = 0.0 if v is None else float(v)
                            examples.append((s_cpu, p_cpu, v_f))
                        # cleanup temp file
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass
                    except Exception as e:
                        worker_errors.append({"worker_id": res.get("worker_id"), "error": f"failed to load examples_file: {e}"})
                else:
                    examples = res.get("examples", [])

                # ensure all tensors are cpu detached (legacy branch)
                safe_examples = []
                for (s, p, v) in examples:
                    try:
                        s_cpu = s.detach().cpu() if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
                    except Exception:
                        s_cpu = torch.tensor(s, dtype=torch.float32)
                    try:
                        p_cpu = p.detach().cpu() if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)
                    except Exception:
                        p_cpu = torch.tensor(p, dtype=torch.float32)
                    v_f = 0.0 if v is None else float(v)
                    safe_examples.append((s_cpu, p_cpu, v_f))

                combined.extend(safe_examples)
                self.log(f"[PAR] Worker {res.get('worker_id')} returned {res.get('count', len(safe_examples))} positions.")
                game_res = res.get("game_results", [])
                game_len = res.get("game_lengths", [])
                all_game_results.extend(game_res)
                all_game_lengths.extend(game_len)

        # join and force-terminate if any remain alive
        for p in processes:
            p.join(timeout=1.0)
        for p in processes:
            if p.is_alive():
                try:
                    self.log(f"[PAR] Terminating unresponsive worker PID={p.pid}")
                    p.terminate()
                    p.join(timeout=1.0)
                except Exception as e:
                    self.log(f"[PAR] Failed to terminate worker PID={getattr(p, 'pid', 'unknown')}: {e}")

        elapsed = time.time() - start_time
        self.log(f"[PAR] Collected total {len(combined)} positions from {num_processes} workers in {elapsed:.1f}s")
        if worker_errors:
            self.log(f"[PAR] {len(worker_errors)} workers reported errors; check earlier logs.")
        return combined, all_game_results, all_game_lengths

    def train_step(self, batch_size=64):
        """
        Train one gradient step on sampled batch.
        Uses distributional cross-entropy for policy (learn full pi distribution),
        and MSE for value head.
        Returns stats dict or None if not enough data.
        """
        if len(self.buffer) < batch_size:
            return None

        states, policies, values = self.buffer.sample(batch_size)
        states = states.to(self.device)
        policies = policies.to(self.device)   # expected shape (B, move_dim), sum to 1
        values = values.to(self.device)       # shape (B,1)

        self.model.train()
        out_policy, out_value = self.model(states)  # out_policy: logits (B, move_dim), out_value: (B,1) or (B,)

        # === Policy loss: distributional cross-entropy ===
        # numerical safety: clamp logits to avoid extreme exponentials
        out_policy = torch.clamp(out_policy, -20.0, 20.0)
        log_probs = torch.log_softmax(out_policy, dim=1)
        loss_p = -(policies * log_probs).sum(dim=1).mean()

        # === Value loss ===
        loss_v = self.value_loss_fn(out_value, values)

        loss = loss_p + loss_v

        # safety: skip if invalid numerics
        if not torch.isfinite(loss):
            self.log("[WARN] Non-finite loss detected, skipping update.")
            return None

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        self.optimizer.step()

        # logging stats: accuracy proxy (argmax against distribution peak)
        with torch.no_grad():
            pred_moves = torch.argmax(out_policy, dim=1)
            true_moves = torch.argmax(policies, dim=1)
            acc = (pred_moves == true_moves).float().mean().item()

        return {"loss": loss.item(), "loss_p": loss_p.item(), "loss_v": loss_v.item(), "acc": acc, "batch": batch_size}

    def save_model(self, suffix="latest"):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        base, ext = os.path.splitext(self.model_path)
        save_path = f"{base}_{suffix}{ext}"
        try:
            torch.save(self.model.state_dict(), save_path)
            # also persist canonical path
            torch.save(self.model.state_dict(), self.model_path)
            # also save optimizer for resume
            try:
                opt_path = self.model_path.replace(".pth", "_opt.pth")
                torch.save(self.optimizer.state_dict(), opt_path)
            except Exception as e:
                self.log(f"[WARN] Could not save optimizer state: {e}")
            self.log(f"[INFO] Model saved ({suffix}) → {save_path}")
        except Exception as e:
            self.log(f"[ERROR] Failed to save model: {e}")

    def _save_selfplay_batch(self, examples):
        """
        Save a collected batch of self-play examples to disk for future analysis / replay.
        examples: list of (state_tensor, policy_tensor, value)
        """
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(SELFPLAY_SAVE_DIR, f"episode_{ts}.pt")
            torch.save({"examples": examples}, fname)
            self.log(f"[IO] Saved self-play batch ({len(examples)} positions) -> {fname}")
        except Exception as e:
            self.log(f"[IO-ERR] Failed saving self-play batch: {e}")

    def _append_stats_csv(self, episode, total_positions, games, avg_game_length, white_rate, black_rate, draw_rate, mean_loss):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(STATS_CSV, "a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow([ts, episode, total_positions, games, avg_game_length, white_rate, black_rate, draw_rate, mean_loss])
        except Exception:
            pass

    def log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} {msg}"
        print(line)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def run_training(self, episodes=50, train_steps=20, batch_size=64, sims=100, verbose=False, collect_in_parallel=True):
        """
        Главен RL loop: for ep -> parallel self-play collection -> add to buffer -> some train steps -> checkpoint
        """
        self.log(f"[RL] Starting training for {episodes} episodes...")
        avg_losses = []

        try:
            for ep in range(1, episodes + 1):
                self.log(f"\n[EP {ep}] Collecting self-play data...")
                if collect_in_parallel:
                    collected, game_results, game_lengths = self._collect_self_play_parallel(num_processes=NUM_PROCESSES,
                                                                                            games_per_worker=GAMES_PER_PROCESS,
                                                                                            sims=sims,
                                                                                            temperature=1.0)
                else:
                    examples, logs = self.self_play_episode(mcts_sims=sims, temperature=1.0, verbose=verbose)
                    collected = examples
                    game_results = []
                    game_lengths = []
                    if collected:
                        game_lengths = [len(collected)]

                # add to global buffer and save raw batch
                if collected:
                    self.add_examples(collected)
                    self._save_selfplay_batch(collected)
                    self.log(f"[EP {ep}] Collected {len(collected)} examples (Buffer size: {len(self.buffer)})")
                else:
                    self.log(f"[EP {ep}] No examples collected this round.")

                # compute aggregated stats
                total_positions = len(collected)
                games = len(game_lengths) if game_lengths else 0
                avg_game_length = float(np.mean(game_lengths)) if game_lengths else 0.0
                white_rate = black_rate = draw_rate = 0.0
                if game_results:
                    total_games = len(game_results)
                    white_rate = 100.0 * sum(1 for r in game_results if r == 1) / total_games
                    black_rate = 100.0 * sum(1 for r in game_results if r == -1) / total_games
                    draw_rate = 100.0 * sum(1 for r in game_results if r == 0) / total_games
                    self.log(f"[STATS] Games:{total_games} Len_avg:{avg_game_length:.2f} White:{white_rate:.1f}% Black:{black_rate:.1f}% Draw:{draw_rate:.1f}%")
                else:
                    self.log(f"[STATS] Collected positions: {total_positions} (no per-game metadata available)")

                self._append_stats_csv(ep, total_positions, games, avg_game_length, white_rate, black_rate, draw_rate, mean_loss=0.0)

                # training
                for step in range(train_steps):
                    stats = self.train_step(batch_size)
                    if stats:
                        self.log(f"[TRAIN] Step {step+1}/{train_steps} | Loss={stats['loss']:.4f} | "
                                 f"Policy={stats['loss_p']:.4f} | Value={stats['loss_v']:.4f} | Acc={stats['acc']*100:.2f}%")
                        avg_losses.append(stats['loss'])
                    else:
                        self.log("[TRAIN] Not enough data or NaN loss.")

                # after training steps, update mean_loss in CSV last row manually (simple append marker)
                if avg_losses:
                    mean_loss = float(np.mean(avg_losses[-max(len(avg_losses), 1):]))
                else:
                    mean_loss = 0.0
                # Append a new CSV row with updated mean_loss for this episode (so there is a row with loss)
                self._append_stats_csv(ep, total_positions, games, avg_game_length, white_rate, black_rate, draw_rate, mean_loss)

                # periodic saving
                if ep % 5 == 0:
                    self.save_model(suffix=f"ep{ep}")

            # final save
            self.save_model(suffix="final")
            self.log("[RL] Training complete!")

        except KeyboardInterrupt:
            self.log("[INTERRUPT] Training stopped manually. Saving checkpoint...")
            self.save_model(suffix="interrupted")
            try:
                opt_path = self.model_path.replace(".pth", "_opt.pth")
                torch.save(self.optimizer.state_dict(), opt_path)
                self.log(f"[INTERRUPT] Optimizer checkpoint saved → {opt_path}")
            except Exception:
                pass

        # final stats
        if avg_losses:
            try:
                self.log(f"[STATS] Mean loss: {np.mean(avg_losses):.4f} over {len(avg_losses)} recorded train steps")
            except Exception:
                pass

# -----------------------
# quick run when executed directly
# -----------------------
if __name__ == "__main__":
    pipeline = RLPipeline(
        model_path="/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth",
        move_vector_size=4672  # adjust if your move encoding differs
    )
    pipeline.run_training(episodes=20, train_steps=15, batch_size=64, sims=400, verbose=True)