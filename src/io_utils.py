import pandas as pd
import numpy as np

def load_csv_numeric(path: str, drop_na: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Loads CSV and keeps only numeric columns.
    Returns: (numeric_df, X ndarray)
    """
    df = pd.read_csv(path)

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found in CSV. PCA needs numeric features.")

    if drop_na:
        numeric_df = numeric_df.dropna(axis=0)

    X = numeric_df.to_numpy(dtype=float)
    return numeric_df, X
