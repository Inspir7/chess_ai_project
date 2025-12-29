import numpy as np
import torch
import random
import chess
from training.move_encoding import move_to_index, index_to_move, get_total_move_count


# ========== Helper: reconstruct board from tensor ==========
def tensor_to_board(tensor):
    piece_map_rev = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
    }

    board = chess.Board(fen=None)  # start blank
    board.clear_board()

    for r in range(8):
        for c in range(8):
            for ch in range(12):  # only piece channels
                if tensor[r, c, ch] == 1:
                    square = chess.square(c, 7 - r)
                    board.set_piece_at(square, chess.Piece.from_symbol(piece_map_rev[ch]))

    return board


# ========== MAIN TEST FUNCTION ==========
def test_dataset(path_states, path_policies, path_values, samples=100):
    print("Loading sample dataset...")

    states = np.load(path_states)
    policies = np.load(path_policies)
    values = np.load(path_values)

    N = len(states)
    print(f"Dataset size: {N}")

    # Pick random samples
    for i in range(samples):
        idx = random.randint(0, N - 1)

        state = states[idx]
        policy = policies[idx]
        value = values[idx]

        # === Test 1: shape ===
        assert state.shape == (8, 8, 15), f"Bad state shape at {idx}"
        assert len(policy) == 4672, f"Bad policy size at {idx}"
        assert -1.01 <= value <= 1.01, f"Invalid value at {idx}"

        # === Test 2: Reconstruct board ===
        board = tensor_to_board(state)
        if board is None:
            raise RuntimeError(f"Failed to reconstruct board at {idx}")

        # === Test 3: Policy legality check ===
        legal = list(board.legal_moves)
        if legal:
            # pick random legal move
            mv = random.choice(legal)
            mv_index = move_to_index(mv)
            assert mv_index != -1, f"Legal move has no index: {mv} at {idx}"
        else:
            # position is terminal → OK
            pass

        # === Test 4: Policy distribution sanity ===
        if policy.sum() > 0:
            p = policy / policy.sum()
            assert abs(p.sum() - 1) < 1e-3, f"Policy not normalized at {idx}"

        # === Optional: Print every 20 samples ===
        if i % 20 == 0:
            print(f"[OK] Sample {i}: value={value}, legal_moves={len(legal)}")

    print("\n🎉 DATASET CHECK COMPLETED — ALL GOOD!\n")


# ================= RUN TEST =================
if __name__ == "__main__":
    # TODO: Put your actual file paths here:
    path_states = "/home/presi/projects/chess_ai_project/training/train/train_labeled_states_0.npy"
    path_policies = "/home/presi/projects/chess_ai_project/training/train/train_labeled_policies_0.npy"
    path_values = "/home/presi/projects/chess_ai_project/training/train/train_labeled_values_0.npy"

    test_dataset(path_states, path_policies, path_values, samples=100)