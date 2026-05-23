"""
Configuration - 경로 및 상수 정의
"""

from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Output paths
OUTPUT_DIR = ROOT / "outputs"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# Model paths
MODEL_DIR = ROOT / "models"
CATBOOST_MODEL = MODEL_DIR / "catboost_model.pkl"
LIGHTGBM_MODEL = MODEL_DIR / "lightgbm_model.pkl"
RULE_MODEL = MODEL_DIR / "rule_model.pkl"
META_MODEL = MODEL_DIR / "meta_model.pkl"

# OOF paths
OOF_DIR = ROOT / "outputs" / "oof"
CATBOOST_OOF = OOF_DIR / "catboost_oof.npy"
LIGHTGBM_OOF = OOF_DIR / "lightgbm_oof.npy"
RULE_OOF = OOF_DIR / "rule_oof.npy"

# Config paths
CONFIG_DIR = ROOT / "configs"

# Constants
N_FOLDS = 5
RANDOM_SEEDS = [41, 42, 43, 44, 45]
TARGET_COL = "completed"
ID_COL = "ID"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
OOF_DIR.mkdir(exist_ok=True, parents=True)
