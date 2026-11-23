"""Problem instance: parameters, practice tables and one farmer panel.

Budgets follow Eq. (46): B_i = A_i * b_bar * (1 + eps_i).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from carbonfarm.config import Params
from carbonfarm.data.panel import generate_panel
from carbonfarm.data.practices import PracticeData, build_practice_data


@dataclass(frozen=True)
class Instance:
    params: Params
    practices: PracticeData
    seed: int
    farm_size: np.ndarray        # (n,) ha
    budget_noise: np.ndarray
    budget_area: np.ndarray      # (n,) reference area for the budget (ha)
    cmp: np.ndarray              # (n, m) tCO2e/ha

    @property
    def n(self) -> int:
        return len(self.farm_size)

    @property
    def budget(self) -> np.ndarray:
        return self.budget_area * self.params.budget_per_ha * (1.0 + self.budget_noise)

    @property
    def avg_farm_size(self) -> float:
        return float(self.farm_size.mean())

    def with_params(self, **changes) -> "Instance":
        return replace(self, params=self.params.replace(**changes))

    def with_budget_per_ha(self, b: float) -> "Instance":
        return self.with_params(budget_per_ha=float(b))

    def with_scaled_farms(self, k: float, scale_budget: bool = True) -> "Instance":
        """Multiply every farm area by k.  scale_budget=True keeps the budget per hectare constant
        (paper Table 3); False keeps the absolute budgets (a pure land-size shock)."""
        return replace(self, farm_size=self.farm_size * k,
                       budget_area=self.budget_area * (k if scale_budget else 1.0))

    def subset(self, idx) -> "Instance":
        idx = np.asarray(list(idx), dtype=int)
        return replace(self, farm_size=self.farm_size[idx], budget_noise=self.budget_noise[idx],
                       budget_area=self.budget_area[idx], cmp=self.cmp[idx])


def make_instance(n: int = 15, seed: int = 42, params: Params | None = None,
                  practices: PracticeData | None = None) -> Instance:
    params = params or Params()
    practices = practices or build_practice_data()
    panel = generate_panel(n, seed, practices, params)
    return Instance(params=params, practices=practices, seed=seed, farm_size=panel.farm_size,
                    budget_noise=panel.budget_noise, budget_area=panel.farm_size.copy(), cmp=panel.cmp)
