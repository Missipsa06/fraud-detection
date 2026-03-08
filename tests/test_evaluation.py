import numpy as np
import pytest

from src.evaluation import find_best_threshold, evaluate_model


@pytest.fixture
def clean_scores():
    """Scores bien séparés : fraudes proches de 1, légitimes proches de 0."""
    np.random.seed(42)
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.concatenate([
        np.random.uniform(0.0, 0.3, 90),
        np.random.uniform(0.7, 1.0, 10),
    ])
    return y_true, y_proba


def test_threshold_is_float(clean_scores):
    y_true, y_proba = clean_scores
    t = find_best_threshold(y_true, y_proba)
    assert isinstance(t, float)


def test_threshold_between_0_and_1(clean_scores):
    y_true, y_proba = clean_scores
    t = find_best_threshold(y_true, y_proba)
    assert 0.0 <= t <= 1.0


def test_threshold_respects_min_precision(clean_scores):
    from src.config import MIN_PRECISION
    from sklearn.metrics import precision_recall_curve

    y_true, y_proba = clean_scores
    t = find_best_threshold(y_true, y_proba)

    y_pred = (y_proba >= t).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    assert precision >= MIN_PRECISION


def test_no_crash_with_inverted_model():
    # Modèle inversé : fraudes ont des scores plus bas que légitimes
    # La fonction doit quand même retourner un float valide
    np.random.seed(0)
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.concatenate([
        np.random.uniform(0.7, 1.0, 90),
        np.random.uniform(0.0, 0.3, 10),
    ])
    t = find_best_threshold(y_true, y_proba)
    assert isinstance(t, float)
    assert 0.0 <= t <= 1.0


def test_evaluate_model_returns_tuple(clean_scores):
    y_true, y_proba = clean_scores
    t = find_best_threshold(y_true, y_proba)
    result = evaluate_model(y_true, y_proba, t)
    assert isinstance(result, tuple) and len(result) == 2


def test_pr_auc_between_0_and_1(clean_scores):
    y_true, y_proba = clean_scores
    t = find_best_threshold(y_true, y_proba)
    _, pr_auc = evaluate_model(y_true, y_proba, t)
    assert 0.0 <= pr_auc <= 1.0


def test_report_has_class_keys(clean_scores):
    y_true, y_proba = clean_scores
    t = find_best_threshold(y_true, y_proba)
    report, _ = evaluate_model(y_true, y_proba, t)
    assert "0" in report and "1" in report
