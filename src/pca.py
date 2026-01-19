import numpy as np

class PCAFromScratch:
    """
    PCA implemented using SVD (stable) and basic linear algebra.
    No scikit-learn PCA used.

    Fit:
      - mean-center X
      - compute SVD of centered matrix
      - principal components = right singular vectors (Vt)
      - explained variance = (S^2) / (n_samples - 1)
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None     # shape: (n_components_max, n_features)
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to compute PCA")

        # 1) mean center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2) SVD of centered data matrix
        # Xc = U S Vt
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        self.singular_values_ = S.copy()
        self.components_ = Vt.copy()

        # 3) explained variance
        # eigenvalues of covariance = (S^2)/(n_samples-1)
        explained_var = (S ** 2) / (n_samples - 1)
        total_var = explained_var.sum()

        self.explained_variance_ = explained_var
        self.explained_variance_ratio_ = explained_var / total_var if total_var > 0 else np.zeros_like(explained_var)

        return self

    def transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        self._check_is_fitted()
        if n_components <= 0:
            raise ValueError("n_components must be >= 1")

        Xc = X - self.mean_
        W = self.components_[:n_components].T  # (n_features, n_components)
        Z = Xc @ W
        return Z

    def inverse_transform(self, Z: np.ndarray, n_components: int) -> np.ndarray:
        self._check_is_fitted()
        W = self.components_[:n_components]  # (n_components, n_features)
        X_recon = Z @ W + self.mean_
        return X_recon

    def reconstruction_error_mse(self, X: np.ndarray, n_components: int) -> float:
        Z = self.transform(X, n_components=n_components)
        Xr = self.inverse_transform(Z, n_components=n_components)
        return float(np.mean((X - Xr) ** 2))

    def _check_is_fitted(self):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAFromScratch is not fitted yet. Call fit(X) first.")
