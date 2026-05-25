# src/pdm/features.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import COLUMNS_TO_DROP, NUMERIC_FEATURES

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused columns and create engineered features.
    Returns a new DataFrame (no in-place modification).
    """
    df = df.copy()

    # Drop unwanted columns (but NOT Air temperature [K])
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Derived features
    df["MPower [W]"] = np.round(
        (df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi) / 60,
        4,
    )
    df["Temperature_difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    return df

def build_preprocessor():
    """
    Build a preprocessing pipeline.
    Here: scale numeric features.
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ]
    )

    return preprocessor