"""Stability of an allocation (Eqs. 28-29, Proposition 1).

A coalition T blocks x if vhat(T) > x(T); its excess is vhat(T) - x(T). Singleton rows are the IR checks.
"""
from __future__ import annotations

import numpy as np


def excess_vector(x: np.ndarray, A: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    return v_hat - A @ x


def stability_metrics(x: np.ndarray, p_grand: float, A: np.ndarray, v_hat: np.ndarray,
                      tol: float = 1e-6) -> dict:
    ex = excess_vector(x, A, v_hat)
    single = A.sum(1) == 1
    eff_gap = float(x.sum() - p_grand)
    return {
        "efficiency_gap": eff_gap,
        "max_excess": float(ex.max()),
        "violations": int((ex > tol).sum()),
        "n_constraints": int(len(ex)),
        "ir_violations": int((ex[single] > tol).sum()),
        "is_stable": bool(ex.max() <= tol and abs(eff_gap) <= 1e-6 * max(1.0, abs(p_grand))),
        "excess": ex,
    }


def nan_metrics() -> dict:
    return {"efficiency_gap": np.nan, "max_excess": np.nan, "violations": np.nan, "n_constraints": np.nan,
            "ir_violations": np.nan, "is_stable": np.nan, "excess": np.array([])}
