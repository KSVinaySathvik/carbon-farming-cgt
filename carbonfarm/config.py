"""Baseline parameters (Table 1), panel settings, seeds and paths.

Params is immutable; derive variants with params.replace(ccp=100).
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

PPP_RATE = 25.0                      # LCU per USD(PPP); only used for the INR -> USD practice tables
PRACTICE_SEED = 42                   # App. B.3.1: interaction matrices drawn once with seed 42
PANEL_SEEDS = tuple(range(42, 52))   # the 10 benchmark farmer panels used for every table
EXACT_MAX_N = 16                     # up to this many farmers all 2^n coalitions are enumerated exactly


@dataclass(frozen=True)
class Params:
    """Economic parameters.  Defaults reproduce Table 1 of the paper."""
    ccp: float = 60.0              # carbon credit price, $/tCO2e                (1500 LCU / 25)
    fix_mrv: float = 2000.0        # Fix_MRV, $                                  (50 000 LCU / 25)
    var_mrv: float = 80.0          # Var_MRV, $/ha^delta                         (2 000 LCU / 25)
    delta: float = 0.7             # MRV scale exponent, in (0, 1)
    fix_t: float = 1000.0          # Fix_T, $                                    (25 000 LCU / 25)
    var_t: float = 10.0            # Var_T, $/farmer                             (250 LCU / 25)
    budget_per_ha: float = 140.0   # b-bar, $/ha
    yield_t_per_ha: float = 4.0    # Y0, conventional yield, t/ha
    crop_price: float = 880.0      # CP, paddy price, $/t                        (22 000 LCU / 25)
    farm_size_min: float = 0.10    # ha  (Sec. 6 says 0.20, App. B.5 says 0.10, Table 6 shows 0.14 ha)
    farm_size_max: float = 2.00    # ha
    lognormal_mu: float = -0.5
    lognormal_sigma: float = 1.0
    budget_noise_sd: float = 0.5   # eps_i ~ N(0, sd^2), clipped to +-budget_noise_clip (Eq. 46)
    budget_noise_clip: float = 0.5
    cmp_noise: float = 0.5
    feas_tol: float = 1e-9         # constraint tolerance in Eqs. (20)-(21)
    stab_tol: float = 1e-6         # tolerance for "core non-empty" / "blocking" tests

    def replace(self, **changes) -> "Params":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    def mrv_cost(self, area_ha):
        return self.fix_mrv + self.var_mrv * np.power(area_ha, self.delta)

    def transaction_cost(self, n_members):
        return self.fix_t + self.var_t * n_members

    def cert_cost(self, area_ha, n_members):
        return self.mrv_cost(area_ha) + self.transaction_cost(n_members)

    @property
    def yield_value_per_pct(self) -> float:
        """Y0 * CP / 100 : $/ha per percentage point of yield change (Eq. 9)."""
        return self.yield_t_per_ha * self.crop_price / 100.0
