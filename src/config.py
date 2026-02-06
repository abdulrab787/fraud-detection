from pathlib import Path

# Project root (fraud-detection/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Paths
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"
SUBMISSION_PATH = PROJECT_ROOT / "submissions" / "submission.csv"
