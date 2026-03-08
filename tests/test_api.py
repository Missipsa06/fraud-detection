import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient

from src.api import app

FAKE_TRANSACTION = {
    "Time": 0.0,
    "Amount": 100.0,
    **{f"V{i}": 0.0 for i in range(1, 29)},
}

FAKE_SAMPLES = {
    "legit": [FAKE_TRANSACTION],
    "fraud": [{**FAKE_TRANSACTION, "Amount": 999.0}],
}


@pytest.fixture
def client():
    mock_model = MagicMock()
    # predict_proba retourne shape (n, 2) → [:, 1] donne [0.1]
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

    with patch("src.api.MODEL_PATH") as mp, \
         patch("src.api.THRESHOLD_PATH") as tp, \
         patch("joblib.load", return_value=mock_model), \
         patch("builtins.open", mock_open()), \
         patch("src.api.json.load", side_effect=[{"threshold": 0.5}, FAKE_SAMPLES]):
        mp.exists.return_value = True
        tp.exists.return_value = True
        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_expected_keys(client):
    r = client.post("/predict", json=FAKE_TRANSACTION)
    assert r.status_code == 200
    data = r.json()
    assert "fraud" in data
    assert "score" in data


def test_predict_score_is_float_between_0_and_1(client):
    r = client.post("/predict", json=FAKE_TRANSACTION)
    score = r.json()["score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_fraud_flag_below_threshold(client):
    # score = 0.1, threshold = 0.5 → pas de fraude
    r = client.post("/predict", json=FAKE_TRANSACTION)
    assert r.json()["fraud"] is False


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in FAKE_TRANSACTION.items() if k != "V1"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_sample_legit(client):
    r = client.get("/sample?fraud=false")
    assert r.status_code == 200
    assert "Amount" in r.json()


def test_sample_fraud(client):
    r = client.get("/sample?fraud=true")
    assert r.status_code == 200
    assert "Amount" in r.json()
