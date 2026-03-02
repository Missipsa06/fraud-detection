import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_amount"] = np.log1p(df["Amount"])
    df["is_round_amount"] = (df["Amount"] == df["Amount"].round(0)).astype(int)

    df["hour"] = (df["Time"] % 86400) // 3600
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)

    return df
