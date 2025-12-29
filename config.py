from pathlib import Path

# ==============================
# PROJECT ROOT (choose ONE)
# ==============================

# LOCAL (Windows)
#PROJECT_ROOT = Path("C:/Users/prezi/PycharmProjects/chess_ai_project")

# REMOTE (WSL / Linux)
PROJECT_ROOT = Path("/home/presi/projects/chess_ai_project")

# ==============================
# DERIVED PATHS
# ==============================

DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
UI_DIR = PROJECT_ROOT / "ui"
CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

def ensure_dirs():
    for d in [
        DATASETS_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)
