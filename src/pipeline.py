import json
import warnings
warnings.filterwarnings("ignore")
import mlflow
import mlflow.sklearn

from .data import load_data, split_data
from .model import train_model, predict_proba
from .evaluation import find_best_threshold, evaluate_model
from .config import MODEL_PARAMS, BEST_PARAMS_PATH


def load_params():
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            print("Paramètres tuning chargés.")
            return json.load(f)
    return MODEL_PARAMS

def run_pipeline():

    df = load_data()
    X_train, X_val, y_train, y_val = split_data(df)
    params = load_params()

    with mlflow.start_run():

        # Log params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        model = train_model(X_train, y_train, params)

        # Predict
        y_proba = predict_proba(model, X_val)

        # Threshold tuning
        threshold = find_best_threshold(y_val, y_proba)
        mlflow.log_param("threshold", threshold)

        # Evaluate
        report, pr_auc = evaluate_model(y_val, y_proba, threshold)

        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("pr_auc", pr_auc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("Recall:", report["1"]["recall"])
        print("Precision:", report["1"]["precision"])
        print("PR-AUC:", pr_auc)
        print("Threshold:", threshold)



if __name__ == "__main__":
    run_pipeline()