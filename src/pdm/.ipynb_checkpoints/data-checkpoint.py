# src/pdm/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import add_derived_features
from .config import TARGET_COL

def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def prepare_data(path: str):
    """
    Load raw CSV, apply feature engineering, and split into X, y.
    """
    df = load_raw_data(path)
    df = add_derived_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y

def split_train_test(X, y, test_size=0.1, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )