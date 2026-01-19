# PCA From Scratch Explorer

A small, test-backed project that implements Principal Component Analysis (PCA) using basic linear algebra (SVD) with **NumPy** — without using scikit-learn’s PCA.

This repo is designed to be:
- **Simple to run** on any numeric CSV
- **Readable** for learning / review
- **Reusable** as a lightweight PCA utility

---

## What this achieves (in plain terms)

Real-world datasets often have many columns that are correlated or redundant. PCA helps you:

- **Reduce dimensionality**: compress many numeric features into a smaller number of “principal components”
- **Preserve information**: keep the directions that explain the most variance in the data
- **Explore structure**: quickly see if the dataset has clusters / patterns (great for visualization)
- **Measure reconstruction quality**: estimate how much information you lose when you keep only `k` components

This repo outputs:
- Explained Variance Ratio (how much each component explains)
- 2D/3D projections (via CLI)
- Reconstruction error (MSE) for a chosen number of components

---

## Project structure

- `src/` — core implementation (PCA + helpers)
- `cli/` — command-line runner
- `tests/` — pytest-based checks to validate behavior
- `data/` — sample inputs you create locally (not required to commit)

---

## Requirements

- Python **3.10+** recommended
- Packages: `numpy`, `pandas` (and `matplotlib` only if you add plots later)

Install deps via `requirements.txt`.

---

## Quickstart (Windows / PowerShell)

### 1) Clone

git clone https://github.com/<your-username>/pca-from-scratch-explorer.git
cd pca-from-scratch-explorer

### 2) Create & activate a virtual environment
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

3) Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Add sample data


Create this file:

Path: data/sample_numeric.csv

Content:

a,b,c,d
1,2,3,4
2,3,4,5
3,4,5,6
4,5,6,7
5,6,7,8
6,7,8,9
7,8,9,10


Note: PCA requires numeric columns. Non-numeric columns are ignored automatically.
