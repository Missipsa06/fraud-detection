from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "class_weight": "balanced",
    "random_state": 42
}

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Business constraint
MIN_PRECISION = 0.4

# Tuning
BEST_PARAMS_PATH = BASE_DIR / "best_params.json"

# Artifacts (modèle sérialisé pour l'API)
ARTIFACTS_DIR  = BASE_DIR / "artifacts"
MODEL_PATH     = ARTIFACTS_DIR / "model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"