# growth/math_utils.py
"""
Mathematical helper functions for linear algebra.
"""
import numpy as np

def _orthonormal_rows(C: np.ndarray) -> np.ndarray:
    u, _, vT = np.linalg.svd(C, full_matrices=False)
    return u @ vT

def _safe_norm(x: np.ndarray) -> float:
    n = float(np.linalg.norm(x))
    return n if n > 1e-12 else 1e-12