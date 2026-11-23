"""Farmer panels (App. B.5).

Farm sizes are truncated log-normal on [0.10, 2.00] ha, budget noise eps_i is a clipped normal
(Params.budget_noise_sd / budget_noise_clip), and CMP_ij = CMP_j * (1 + U(-0.2, 0.2)). The panel stores the budget noise
rather than the budget so that budget sweeps keep each farmer's relative endowment.

    python -m carbonfarm.data.panel --n 15 --seed 42
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carbonfarm.config import Params
from carbonfarm.data.practices import PracticeData


@dataclass(frozen=True)
class Panel:
    farm_size: np.ndarray      # (n,)   ha
    budget_noise: np.ndarray   # (n,)   eps_i in [-0.2, 0.2]
    cmp: np.ndarray            # (n, m) tCO2e/ha, farmer-specific CMP_ij


def truncated_lognormal(rng, n, mu, sigma, low, high, decimals=2) -> np.ndarray:
    vals: list = []
    while len(vals) < n:
        x = rng.lognormal(mean=mu, sigma=sigma, size=n * 2)
        vals.extend(x[(x >= low) & (x <= high)].tolist())
    return np.round(np.clip(np.array(vals[:n]), low, high), decimals)


def generate_panel(n: int, seed: int, practices: PracticeData, params: Params | None = None) -> Panel:
    p = params or Params()
    rng = np.random.default_rng(seed)
    sizes = truncated_lognormal(rng, n, p.lognormal_mu, p.lognormal_sigma, p.farm_size_min, p.farm_size_max)
    noise = np.clip(rng.normal(0.0, p.budget_noise_sd, size=n), -p.budget_noise_clip, p.budget_noise_clip)
    csp_noise = rng.uniform(-p.cmp_noise, p.cmp_noise, size=(n, practices.m))
    return Panel(sizes, noise, practices.cmp_base[None, :] * (1.0 + csp_noise))


if __name__ == "__main__":
    import argparse
    from carbonfarm.data.practices import build_practice_data
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=15); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    pn = generate_panel(a.n, a.seed, build_practice_data())
    for i, (s, e) in enumerate(zip(pn.farm_size, pn.budget_noise)):
        print(f"F{i+1:<3d} size={s:5.2f} ha  budget_noise={e:+.3f}")
