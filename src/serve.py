"""
Script à lancer une fois pour entraîner et sauvegarder le modèle.

Usage:
    python -m src.serve
"""
import json
import joblib

from .data import load_data, split_data
from .model import train_model, predict_proba
from .evaluation import find_best_threshold
from .pipeline import load_params
from .config import ARTIFACTS_DIR, MODEL_PATH, THRESHOLD_PATH


def save_artifacts():
    print("Chargement des données...")
    df = load_data()
    X_train, X_val, y_train, y_val = split_data(df)

    params = load_params()
    print("Entraînement du modèle...")
    model = train_model(X_train, y_train, params)

    y_proba = predict_proba(model, X_val)
    threshold = find_best_threshold(y_val, y_proba)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold}, f)

    print(f"Modèle sauvegardé      : {MODEL_PATH}")
    print(f"Threshold sauvegardé   : {THRESHOLD_PATH}  ({threshold:.4f})")


if __name__ == "__main__":
    save_artifacts()
