import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, TEST_SIZE, RANDOM_STATE
from .features import build_features


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
