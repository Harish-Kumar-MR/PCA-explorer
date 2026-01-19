# PCA From Scratch Explorer

A small project that implements PCA using basic linear algebra (SVD) with NumPy.
This is designed as a portfolio-grade, reusable utility for exploring any numeric dataset.

## Features
- PCA from scratch (no scikit-learn PCA)
- Explained variance ratio
- 2D/3D projections via CLI
- Reconstruction error (MSE) vs number of components
- Numeric CSV support (auto-select numeric columns)

## Setup
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Pytest uses tests/conftest.py to add repo root to sys.path for imports.