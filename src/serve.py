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
from .config import ARTIFACTS_DIR, MODEL_PATH, THRESHOLD_PATH, SAMPLES_PATH

COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


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

    # Sauvegarder quelques exemples pour le endpoint /sample
    import pandas as pd
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_proba_s = pd.Series(y_proba, index=X_val.index)

    # Légitimes : les 5 avec le score le plus bas (plus sûrement légitimes)
    legit_idx = y_proba_s[y_val == 0].nsmallest(5).index.tolist()
    # Fraudes : les 5 avec le score le plus élevé (mieux détectées)
    fraud_idx = y_proba_s[y_val == 1].nlargest(5).index.tolist()
    samples = {
        "legit": X_val.loc[legit_idx, COLS].to_dict(orient="records"),
        "fraud": X_val.loc[fraud_idx, COLS].to_dict(orient="records"),
    }
    with open(SAMPLES_PATH, "w") as f:
        json.dump(samples, f)

    print(f"Modèle sauvegardé      : {MODEL_PATH}")
    print(f"Threshold sauvegardé   : {THRESHOLD_PATH}  ({threshold:.4f})")
    print(f"Exemples sauvegardés   : {SAMPLES_PATH}")


if __name__ == "__main__":
    save_artifacts()
