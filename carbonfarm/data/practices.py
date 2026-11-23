"""Practice data (App. B.1-B.3).

For the 8 practices: base sequestration CMP_j, operating cost OC_j ($/ha, PPP-adjusted), yield change
Y_j (%), the pairwise interaction matrices alpha (carbon), beta (cost), gamma (yield) and the
incompatibility matrix Delta. Interactions are drawn once with seed 42 from the distributions of
Table 12, using the pair categories of Table 13.

    python -m carbonfarm.data.practices    # writes data/Dataset_USD/*.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from carbonfarm.config import DATA_DIR, PPP_RATE, PRACTICE_SEED

CRR = "Crop Residue Retention"
RMT = "Reduced / Minimum Tillage"
AWD = "Alternate Wetting and Drying (AWD)"
SRI = "System of Rice Intensification (SRI)"
DSR = "Direct Seeded Rice (DSR)"
INM = "Integrated Nutrient Management (INM) – Organic + Inorganic"
MA = "Manure Application"
BA = "Biochar Application"

SHORT_NAMES = {CRR: "CRR", RMT: "RMT", AWD: "AWD", SRI: "SRI", DSR: "DSR", INM: "INM", MA: "MA", BA: "BA"}

_BASE_INR = {
    CRR: (0.60, -395, +5.2), RMT: (1.17, -1250, +4.8), AWD: (2.87, -550, 0.0), SRI: (2.18, 1500, +15.0),
    DSR: (2.89, 0, -12.0), INM: (1.83, 1200, +1.4), MA: (1.69, 3000, +3.9), BA: (2.20, 20000, +10.7),
}

# Pair categories (Table 13).  Pairs not listed are "zero_like".
_ALPHA = {"strong_pos": [(BA, CRR), (BA, MA), (CRR, RMT), (AWD, DSR)],
          "pos": [(CRR, MA), (CRR, INM), (CRR, DSR), (AWD, BA), (BA, INM)],
          "neg": [(DSR, RMT), (DSR, MA), (INM, MA)],
          "neg_strong": [(INM, SRI), (MA, SRI), (BA, SRI)]}
_GAMMA = {"strong_pos": [(BA, MA), (CRR, INM), (AWD, BA), (BA, INM)],
          "pos": [(BA, CRR), (CRR, RMT), (CRR, MA), (AWD, DSR)],
          "neg": [(CRR, DSR), (DSR, RMT), (DSR, MA), (INM, MA)],
          "neg_strong": [(INM, SRI), (MA, SRI), (BA, SRI)]}
_BETA = {"strong_pos": [(AWD, DSR), (DSR, RMT)],
         "pos": [(AWD, RMT), (CRR, RMT)],
         "neg": [(BA, MA), (BA, INM), (CRR, DSR), (INM, MA)],
         "neg_strong": [(INM, SRI), (MA, SRI), (BA, SRI)]}
_HARD_INCOMPATIBLE = [(DSR, SRI), (AWD, SRI), (CRR, SRI), (RMT, SRI)]            # Table 14

_DIST_ALPHA = {"strong_pos": (0.12, 0.03), "pos": (0.08, 0.03), "neg": (-0.15, 0.04),
               "neg_strong": (-0.25, 0.05), "zero_like": (0.0, 0.015)}
_DIST_GAMMA = {"strong_pos": (0.75, 0.375), "pos": (0.375, 0.25), "neg": (-2.25, 0.75),
               "neg_strong": (-7.0, 0.5), "zero_like": (0.0, 0.25)}
_DIST_BETA_INR = {"strong_pos": (-1200.0, 250.0), "pos": (-500.0, 150.0), "neg": (1200.0, 300.0),
                  "neg_strong": (2000.0, 400.0), "zero_like": (0.0, 100.0)}
_CLIP_ALPHA, _CLIP_GAMMA, _CLIP_BETA_INR = (-0.30, 0.30), (-7.50, 3.75), (-2000.0, 1500.0)


@dataclass(frozen=True)
class PracticeData:
    names: tuple
    cmp_base: np.ndarray      # (m,)   tCO2e/ha
    oc: np.ndarray            # (m,)   $/ha   (negative = saving)
    yield_pct: np.ndarray     # (m,)   %
    alpha: np.ndarray         # (m,m)  carbon interaction
    beta: np.ndarray          # (m,m)  cost interaction, $/ha
    gamma: np.ndarray         # (m,m)  yield interaction, pct-points
    incompat: np.ndarray      # (m,m)  int, 1 = hard-incompatible

    @property
    def m(self) -> int:
        return len(self.names)

    @property
    def short(self) -> tuple:
        return tuple(SHORT_NAMES.get(nm, nm[:3].upper()) for nm in self.names)

    def incompatible_pairs(self):
        return [(j, k) for j in range(self.m) for k in range(j + 1, self.m) if self.incompat[j, k] == 1]

    def practice_table(self) -> pd.DataFrame:
        return pd.DataFrame({"Practice": self.names, "Net_CSP_base": self.cmp_base,
                             "Net_OC_per_ha_USD": self.oc, "Yield_change_percent": self.yield_pct})


def _category_lookup(table: dict) -> dict:
    out = {}
    for cat, pairs in table.items():
        for a, b in pairs:
            out[tuple(sorted((a, b)))] = cat
    return out


@lru_cache(maxsize=8)
def build_practice_data(seed: int = PRACTICE_SEED, ppp: float = PPP_RATE) -> PracticeData:
    """Practice tables built in memory. Seed 42 reproduces data/Dataset_USD exactly."""
    names = tuple(_BASE_INR)
    m = len(names)
    idx = {p: i for i, p in enumerate(names)}
    cat_a, cat_g, cat_b = map(_category_lookup, (_ALPHA, _GAMMA, _BETA))
    clip_b = (_CLIP_BETA_INR[0] / ppp, _CLIP_BETA_INR[1] / ppp)

    rng = np.random.default_rng(seed)
    alpha, gamma, beta = (np.zeros((m, m)) for _ in range(3))
    for p1, p2 in combinations(names, 2):
        key = tuple(sorted((p1, p2)))
        i, j = idx[p1], idx[p2]
        ma, sa = _DIST_ALPHA[cat_a.get(key, "zero_like")]
        mg, sg = _DIST_GAMMA[cat_g.get(key, "zero_like")]
        mb, sb = _DIST_BETA_INR[cat_b.get(key, "zero_like")]
        a = float(np.clip(rng.normal(ma, sa), *_CLIP_ALPHA))
        g = float(np.clip(rng.normal(mg, sg), *_CLIP_GAMMA))
        b = float(np.clip(rng.normal(mb / ppp, sb / ppp), *clip_b))
        alpha[i, j] = alpha[j, i] = a
        gamma[i, j] = gamma[j, i] = g
        beta[i, j] = beta[j, i] = b

    incompat = np.zeros((m, m), dtype=int)
    for p1, p2 in _HARD_INCOMPATIBLE:
        i, j = idx[p1], idx[p2]
        incompat[i, j] = incompat[j, i] = 1
        alpha[i, j] = alpha[j, i] = gamma[i, j] = gamma[j, i] = beta[i, j] = beta[j, i] = 0.0

    assert all(np.allclose(M, M.T) for M in (alpha, beta, gamma)) and np.array_equal(incompat, incompat.T)
    return PracticeData(
        names=names,
        cmp_base=np.array([_BASE_INR[p][0] for p in names], float),
        oc=np.array([_BASE_INR[p][1] / ppp for p in names], float),
        yield_pct=np.array([_BASE_INR[p][2] for p in names], float),
        alpha=alpha, beta=beta, gamma=gamma, incompat=incompat)


def write_practice_csvs(pdata: PracticeData, out_dir=None) -> Path:
    out = Path(out_dir) if out_dir else DATA_DIR / "Dataset_USD"
    out.mkdir(parents=True, exist_ok=True)
    pdata.practice_table().to_csv(out / "practices.csv", index=False)
    for fname, M in [("alpha_carbon.csv", pdata.alpha), ("gamma_yield.csv", pdata.gamma),
                     ("beta_cost_usd.csv", pdata.beta), ("delta_incompatibility.csv", pdata.incompat)]:
        pd.DataFrame(M, index=pdata.names, columns=pdata.names).to_csv(out / fname)
    return out


def load_practice_csvs(data_dir) -> PracticeData:
    """Read the practice tables from CSV."""
    d = Path(data_dir)
    for need in ("practices.csv", "alpha_carbon.csv", "beta_cost_usd.csv", "gamma_yield.csv",
                 "delta_incompatibility.csv"):
        if not (d / need).exists():
            raise FileNotFoundError(f"{need} missing in {d}; run `python -m carbonfarm.data.practices`")
    pr = pd.read_csv(d / "practices.csv")
    names = tuple(pr["Practice"])
    mat = lambda f, dt=float: pd.read_csv(d / f, index_col=0).reindex(index=names, columns=names).to_numpy(dt)
    return PracticeData(names, pr["Net_CSP_base"].to_numpy(float), pr["Net_OC_per_ha_USD"].to_numpy(float),
                        pr["Yield_change_percent"].to_numpy(float), mat("alpha_carbon.csv"),
                        mat("beta_cost_usd.csv"), mat("gamma_yield.csv"), mat("delta_incompatibility.csv", int))


if __name__ == "__main__":
    print("wrote", write_practice_csvs(build_practice_data()))
