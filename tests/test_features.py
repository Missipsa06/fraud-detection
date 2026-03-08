import numpy as np
import pandas as pd
import pytest

from src.features import build_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Time": [0.0, 46800.0, 3600.0, 7200.0],
        "Amount": [100.0, 50.0, 33.57, 200.00],
        "V1": [0.1, 0.2, 0.3, 0.4],
    })


def test_new_columns_present(sample_df):
    result = build_features(sample_df)
    for col in ["log_amount", "is_round_amount", "hour", "is_night"]:
        assert col in result.columns


def test_log_amount(sample_df):
    result = build_features(sample_df)
    expected = np.log1p(sample_df["Amount"])
    pd.testing.assert_series_equal(result["log_amount"], expected, check_names=False)


def test_is_round_amount(sample_df):
    result = build_features(sample_df)
    # 100.0, 50.0, 200.00 sont ronds ; 33.57 ne l'est pas
    assert result["is_round_amount"].tolist() == [1, 1, 0, 1]


def test_hour_extraction(sample_df):
    result = build_features(sample_df)
    # Time=0 → 0h, Time=46800 → 13h, Time=3600 → 1h, Time=7200 → 2h
    assert result["hour"].tolist() == [0, 13, 1, 2]


def test_is_night(sample_df):
    result = build_features(sample_df)
    # hour 0, 13, 1, 2 → nuit: 1, 0, 1, 1
    assert result["is_night"].tolist() == [1, 0, 1, 1]


def test_no_mutation(sample_df):
    original_cols = list(sample_df.columns)
    build_features(sample_df)
    assert list(sample_df.columns) == original_cols
