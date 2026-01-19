import argparse
import numpy as np
from src.io_utils import load_csv_numeric
from src.preprocess import standardize
from src.pca import PCAFromScratch

def main():
    parser = argparse.ArgumentParser(description="PCA From Scratch Explorer (CLI)")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--components", type=int, default=2, help="Number of components to project onto")
    parser.add_argument("--standardize", action="store_true", help="Standardize features before PCA")
    args = parser.parse_args()

    numeric_df, X = load_csv_numeric(args.csv)
    if args.standardize:
        X, mean, std = standardize(X)

    pca = PCAFromScratch().fit(X)

    k = args.components
    Z = pca.transform(X, n_components=k)
    mse = pca.reconstruction_error_mse(X, n_components=k)

    print("\n=== PCA From Scratch Explorer ===")
    print(f"Input file: {args.csv}")
    print(f"Rows used: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Components requested: {k}")

    evr = pca.explained_variance_ratio_
    print("\nExplained Variance Ratio (first 10):")
    for i in range(min(10, len(evr))):
        print(f"  PC{i+1}: {evr[i]:.4f}")

    print(f"\nReconstruction MSE with k={k}: {mse:.6f}")

    print("\nFirst 5 projected rows:")
    np.set_printoptions(suppress=True, precision=4)
    print(Z[:5])

if __name__ == "__main__":
    main()
