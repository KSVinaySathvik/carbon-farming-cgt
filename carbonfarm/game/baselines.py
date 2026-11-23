"""Revenue-sharing baselines (Sec. 4.7.1): area-proportional (Eq. 33) and attributable-revenue (Eq. 32)."""
from __future__ import annotations

import numpy as np


def area_proportional(farm_size: np.ndarray, p_grand: float) -> np.ndarray:
    return farm_size / farm_size.sum() * p_grand


def attributable_revenue(R: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    return R - kappa
