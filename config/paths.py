from pathlib import Path

"""
LOCAL USAGE:
  - uncomment LOCAL path
REMOTE USAGE:
  - uncomment REMOTE path
DO NOT COMMIT THIS FILE
"""

# ==============================
# PROJECT ROOT (choose ONE)
# ==============================

# LOCAL (Windows)
PROJECT_ROOT = Path("C:/Users/prezi/PycharmProjects/chess_ai_project")

# REMOTE (WSL / Linux)
# PROJECT_ROOT = Path("/home/presi/projects/chess_ai_project")

assert PROJECT_ROOT.exists(), f"Invalid PROJECT_ROOT: {PROJECT_ROOT}"

# ==============================
# DERIVED PATHS
# ==============================

DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
GUI_DIR = PROJECT_ROOT / "gui"
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