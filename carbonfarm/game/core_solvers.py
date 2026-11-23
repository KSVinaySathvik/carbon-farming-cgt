"""Least core, nucleolus and core geometry on the deviation game vhat (Eqs. 27-30).

Efficiency: sum_i x_i = p(F). Stability rows: x(T) >= vhat(T), with |T| = 1 rows giving IR.
"""
from __future__ import annotations

import warnings

import numpy as np
from carbonfarm.game import backend


def _scale(p, b) -> float:
    return float(max(1.0, abs(p), np.max(np.abs(b)) if len(b) else 1.0))


def _lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    return backend.lp(c, A_ub, b_ub, A_eq, b_eq)


# least core
def least_core_epsilon(p: float, A: np.ndarray, b: np.ndarray):
    """Step 1 of Eq. (30).  Returns (eps*, a least-core point x)."""
    M, n = A.shape
    s = _scale(p, b)
    c = np.zeros(n + 1); c[-1] = 1.0
    res = _lp(c, A_ub=np.hstack([-A, -np.ones((M, 1))]), b_ub=-b / s,
              A_eq=np.hstack([np.ones((1, n)), np.zeros((1, 1))]), b_eq=[p / s])
    if res.status != 0:
        raise RuntimeError(f"least-core LP failed: {res.message}")
    return float(res.x[-1] * s), res.x[:n] * s


def _min_norm_point(p, A, b, eps, x_fallback, slack=1e-7):
    s = _scale(p, b)
    x = backend.qp_min_norm(A, (b - eps) / s - slack, p / s)
    if x is None:
        warnings.warn("min-norm QP failed; returning the LP least-core point")
        return x_fallback
    return x * s


def least_core(p: float, A: np.ndarray, b: np.ndarray, stab_tol: float = 1e-6) -> dict:
    eps, x_lp = least_core_epsilon(p, A, b)
    x = _min_norm_point(p, A, b, eps, x_lp)
    return {"epsilon_star": eps, "allocation": x, "core_nonempty": bool(eps <= stab_tol),
            "lp_point": x_lp}


# nucleolus
def nucleolus(p: float, A: np.ndarray, b: np.ndarray, max_stages: int = 400, dual_tol: float = 1e-9) -> dict:
    M, n = A.shape
    s = _scale(p, b)
    bb, pp = b / s, p / s
    active = np.ones(M, dtype=bool)
    pinned_A, pinned_rhs = [], []
    eps_seq, x = [], None
    c = np.zeros(n + 1); c[-1] = 1.0
    for stage in range(max_stages):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        A_eq = [np.append(np.ones(n), 0.0)] + [np.append(r, 0.0) for r in pinned_A]
        b_eq = [pp] + list(pinned_rhs)
        res = _lp(c, A_ub=np.hstack([-A[idx], -np.ones((idx.size, 1))]), b_ub=-bb[idx],
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq))
        if res.status != 0:
            warnings.warn(f"nucleolus stage {stage}: LP status {res.status} ({res.message})")
            break
        eps = float(res.x[-1]); x = res.x[:n]
        eps_seq.append(eps * s)
        duals = -np.asarray(res.ineqlin.marginals)
        fixed = idx[duals > dual_tol]
        if fixed.size == 0:                                                   # numerically degenerate
            fixed = idx[[int(np.argmax(duals))]]
        for r in fixed:
            pinned_A.append(A[r].copy()); pinned_rhs.append(bb[r] - eps)
        active[fixed] = False
        if np.linalg.matrix_rank(np.vstack([np.ones(n)] + pinned_A), tol=1e-9) >= n:
            break                                                             # x is unique
    return {"allocation": x * s, "n_stages": len(eps_seq), "epsilon_sequence": eps_seq}


# core geometry
def core_intervals(p: float, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 2) array of [min x_i, max x_i] over the core {x(N)=p, x(T) >= vhat(T)}; NaN if empty."""
    M, n = A.shape
    s = _scale(p, b)
    out = np.full((n, 2), np.nan)
    for i in range(n):
        for col, sign in ((0, 1.0), (1, -1.0)):
            c = np.zeros(n); c[i] = sign
            res = _lp(c, A_ub=-A, b_ub=-b / s, A_eq=np.ones((1, n)), b_eq=[p / s])
            if res.status == 0:
                out[i, col] = sign * res.fun * s
    return out


def _null_basis(n: int) -> np.ndarray:
    """Orthonormal basis (n, n-1) of {z : sum z = 0}."""
    Q, _ = np.linalg.qr(np.hstack([np.ones((n, 1)) / np.sqrt(n), np.eye(n)[:, : n - 1]]))
    return Q[:, 1:]


def core_centroid(p: float, A: np.ndarray, b: np.ndarray, x0: np.ndarray, rng: np.random.Generator,
                  n_steps: int = 40000, adapt_rounds: int = 3, adapt_steps: int = 2000,
                  n_batches: int = 10) -> dict:
    """Centre of mass of the core by (adaptive) hit-and-run.

    x0 must be strictly interior (e.g. the least-core point when eps* < 0).  The core is typically very
    elongated, so plain hit-and-run mixes slowly; during the first `adapt_rounds * adapt_steps` steps the
    sampler re-estimates the covariance of the visited points and uses it to shape the (symmetric)
    random directions.  The covariance is then FROZEN, so the production chain is an ordinary
    reversible hit-and-run chain with the uniform distribution on the core as its stationary law.
    Returns the sample mean and a batch-means standard error."""
    M, n = A.shape
    s = _scale(p, b)
    Z = _null_basis(n)
    AZ = A @ Z
    bs = b / s
    x = x0 / s
    slack = A @ x - bs
    if slack.min() <= 0:
        raise ValueError("starting point is not strictly inside the core")
    L = np.eye(n - 1)
    sums = np.zeros((n_batches, n)); counts = np.zeros(n_batches)
    batch_len = max(1, n_steps // n_batches)
    hist = []
    total_steps = adapt_rounds * adapt_steps + n_steps
    for step in range(total_steps):
        u = L @ rng.standard_normal(n - 1)
        u /= np.linalg.norm(u)
        Ad = AZ @ u
        with np.errstate(divide="ignore", invalid="ignore"):
            r = slack / Ad
        hi = np.min(np.where(Ad < -1e-13, -r, np.inf))
        lo = np.max(np.where(Ad > 1e-13, -r, -np.inf))
        t = rng.uniform(lo, hi)
        x = x + t * (Z @ u)
        slack = slack + t * Ad
        if step % 2000 == 1999:
            slack = A @ x - bs                                                # remove round-off drift
        if step < adapt_rounds * adapt_steps:
            hist.append(Z.T @ x)
            if len(hist) == adapt_steps:                                      # end of an adaptation round
                cov = np.cov(np.array(hist).T) + 1e-12 * np.eye(n - 1)
                L = np.linalg.cholesky(cov)
                hist = []
        else:
            k = min((step - adapt_rounds * adapt_steps) // batch_len, n_batches - 1)
            sums[k] += x; counts[k] += 1
    means = sums / counts[:, None]
    return {"allocation": means.mean(0) * s,
            "stderr": means.std(0, ddof=1) / np.sqrt(n_batches) * s if n_batches > 1 else np.zeros(n)}


def analytic_center(p: float, A: np.ndarray, b: np.ndarray, x0: np.ndarray, iters: int = 100) -> np.ndarray:
    """Maximiser of sum_T log(x(T) - vhat(T)) on {x(N) = p} (damped Newton).  x0 strictly interior."""
    M, n = A.shape
    s = _scale(p, b)
    Z = _null_basis(n)
    AZ = A @ Z
    x = x0 / s
    bs = b / s
    f = lambda sl: -np.sum(np.log(sl))
    for _ in range(iters):
        sl = A @ x - bs
        grad = -AZ.T @ (1.0 / sl)
        H = (AZ.T * (1.0 / sl ** 2)) @ AZ
        dz = -np.linalg.solve(H, grad)
        dec = float(-grad @ dz)
        if dec < 1e-14:
            break
        step, f0 = 1.0, f(sl)
        d = Z @ dz
        while step > 1e-12:
            sl_new = sl + step * (A @ d)
            if sl_new.min() > 0 and f(sl_new) <= f0 - 0.25 * step * dec:
                break
            step *= 0.5
        x = x + step * d
    return x * s
