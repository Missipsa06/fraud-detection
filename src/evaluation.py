import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score
from .config import MIN_PRECISION

def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision a len(thresholds)+1 éléments : le dernier est un point d'ancrage
    # sklearn sans threshold associé — on l'exclut
    valid_idx = np.where(precision[:-1] >= MIN_PRECISION)[0]

    if len(valid_idx) == 0:
        return 0.5  # fallback

    best_idx = valid_idx[np.argmax(recall[valid_idx])]
    return float(thresholds[best_idx])

def evaluate_model(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True)
    pr_auc = average_precision_score(y_true, y_proba)

    return report, pr_auc