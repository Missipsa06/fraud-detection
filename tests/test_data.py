import pandas as pd
import pytest

from src.data import split_data
from src.config import TEST_SIZE


@pytest.fixture
def sample_df():
    """DataFrame synthétique avec les colonnes minimales attendues."""
    n = 100
    data = {
        "V1": range(n),
        "Amount": [float(i) for i in range(n)],
        "Time": [float(i * 3600) for i in range(n)],
        "Class": [0] * 90 + [1] * 10,
    }
    return pd.DataFrame(data)


def test_split_returns_four_elements(sample_df):
    result = split_data(sample_df)
    assert len(result) == 4


def test_val_size(sample_df):
    X_train, X_val, _, _ = split_data(sample_df)
    total = len(X_train) + len(X_val)
    assert abs(len(X_val) / total - TEST_SIZE) < 0.05


def test_class_column_absent(sample_df):
    X_train, X_val, _, _ = split_data(sample_df)
    assert "Class" not in X_train.columns
    assert "Class" not in X_val.columns


def test_stratification(sample_df):
    _, _, y_train, y_val = split_data(sample_df)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_val.unique()) == {0, 1}
