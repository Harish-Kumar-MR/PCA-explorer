import numpy as np
from src.pca import PCAFromScratch

def test_shapes_and_variance_ratio_sum():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))

    pca = PCAFromScratch().fit(X)
    assert pca.components_.shape == (5, 5)

    s = float(np.sum(pca.explained_variance_ratio_))
    assert 0.999 <= s <= 1.001

def test_reconstruction_error_decreases_with_more_components():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))

    pca = PCAFromScratch().fit(X)
    mse1 = pca.reconstruction_error_mse(X, n_components=1)
    mse3 = pca.reconstruction_error_mse(X, n_components=3)
    mse6 = pca.reconstruction_error_mse(X, n_components=6)

    assert mse3 <= mse1 + 1e-9
    assert mse6 <= mse3 + 1e-9
