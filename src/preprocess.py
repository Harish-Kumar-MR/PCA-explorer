import numpy as np

def standardize(X: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features: (X - mean) / std
    Returns: X_std, mean, std
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std < eps, 1.0, std)
    X_std = (X - mean) / std
    return X_std, mean, std

def mean_center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean center features: X - mean
    Returns: X_centered, mean
    """
    mean = X.mean(axis=0)
    return X - mean, mean
