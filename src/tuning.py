import json
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier

from .data import load_data, split_data
from .config import RANDOM_STATE, BEST_PARAMS_PATH

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)

        y_proba = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, y_proba))

    return sum(scores) / len(scores)


def run_tuning(n_trials: int = 30):
    print("Chargement des données...")
    df = load_data()
    X_train, _, y_train, _ = split_data(df)

    print(f"Lancement du tuning ({n_trials} essais)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_params = study.best_params
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = RANDOM_STATE

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Meilleur PR-AUC (CV) : {study.best_value:.4f}")
    print(f"Meilleurs paramètres sauvegardés dans : {BEST_PARAMS_PATH}")
    print(best_params)


if __name__ == "__main__":
    run_tuning()
