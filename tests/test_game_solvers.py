"""Allocation solvers on games with known answers (Talmud game, 2-player core, empty core,
Shapley axioms, nucleolus optimality).
"""
import itertools

import numpy as np

from carbonfarm.game.core_solvers import (analytic_center, core_centroid, core_intervals, least_core,
                                          nucleolus)
from carbonfarm.game.shapley import shapley_exact


def _game_from_function(n, f):
    """Return (V over all masks, A rows for proper coalitions, v on those rows)."""
    V = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        V[mask] = f([i for i in range(n) if mask >> i & 1])
    masks = np.arange(1, (1 << n) - 1)
    A = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
    return V, A, V[masks]


def bankruptcy(E, claims):
    claims = np.array(claims, float)
    return lambda S: max(0.0, E - claims.sum() + claims[S].sum()) if len(S) < len(claims) else E


def test_nucleolus_talmud():
    for E, expect in [(100, [100 / 3] * 3), (200, [50, 75, 75]), (300, [50, 100, 150])]:
        V, A, v = _game_from_function(3, bankruptcy(E, [100, 200, 300]))
        nuc = nucleolus(float(E), A, v)["allocation"]
        assert np.allclose(nuc, expect, atol=1e-6), (E, nuc, expect)
    print("nucleolus: Talmud bankruptcy game OK")


def test_shapley_properties():
    V, A, v = _game_from_function(3, bankruptcy(200, [100, 200, 300]))
    phi = shapley_exact(V, 3)
    assert abs(phi.sum() - 200) < 1e-9
    assert np.allclose(phi, [100 / 3, 250 / 3, 250 / 3]), phi                  # known Talmud Shapley value
    # symmetric players get equal value; dummy gets 0
    f = lambda S: 10.0 * len([i for i in S if i < 2]) ** 2 if len(S) else 0.0              # player 2 is a dummy
    V2, _, _ = _game_from_function(3, f)
    phi2 = shapley_exact(V2, 3)
    assert abs(phi2[2]) < 1e-12 and abs(phi2[0] - phi2[1]) < 1e-12 and abs(phi2.sum() - V2[-1]) < 1e-9
    print("Shapley: efficiency / symmetry / dummy OK")


def test_core_two_player_and_centre():
    V, A, v = _game_from_function(2, lambda S: 1.0 if S == [0] else 2.0 if S == [1] else 6.0)
    lc = least_core(6.0, A, v)
    assert lc["epsilon_star"] < 0 and lc["core_nonempty"]
    iv = core_intervals(6.0, A, v)
    assert np.allclose(iv, [[1, 4], [2, 5]], atol=1e-7), iv
    ac = analytic_center(6.0, A, v, x0=lc["lp_point"])
    assert np.allclose(ac, [2.5, 3.5], atol=1e-6), ac
    cc = core_centroid(6.0, A, v, x0=lc["lp_point"], rng=np.random.default_rng(0), n_steps=20000)
    assert np.allclose(cc["allocation"], [2.5, 3.5], atol=0.08), cc["allocation"]
    print("core intervals / analytic centre / hit-and-run centroid OK")


def test_empty_core():
    # 3-player majority game: v(S)=1 if |S|>=2.  Core empty; least core eps* = 1/3, LC point = (1/3,1/3,1/3)
    V, A, v = _game_from_function(3, lambda S: 1.0 if len(S) >= 2 else 0.0)
    lc = least_core(1.0, A, v)
    assert abs(lc["epsilon_star"] - 1 / 3) < 1e-7 and not lc["core_nonempty"], lc
    assert np.allclose(lc["allocation"], [1 / 3] * 3, atol=1e-6)
    assert np.allclose(nucleolus(1.0, A, v)["allocation"], [1 / 3] * 3, atol=1e-6)
    print("empty-core majority game OK (eps* = 1/3)")


def test_nucleolus_is_lexicographically_minimal_on_random_game():
    rng = np.random.default_rng(3)
    n = 5
    base = rng.uniform(0, 10, 1 << n); base[0] = 0
    V = np.zeros(1 << n)
    for mask in range(1, 1 << n):                                   # superadditive-ish random game
        members = [i for i in range(n) if mask >> i & 1]
        V[mask] = base[mask] + 3.0 * len(members) ** 1.5
    masks = np.arange(1, (1 << n) - 1)
    A = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
    v, p = V[masks], V[-1]
    x = nucleolus(p, A, v)["allocation"]
    assert abs(x.sum() - p) < 1e-7
    key = lambda y: tuple(np.sort(v - A @ y)[::-1].round(7))
    kx = key(x)
    for _ in range(3000):
        d = rng.standard_normal(n); d -= d.mean()
        for scale in (1e-3, 1e-2, 1e-1, 1.0):
            assert key(x + scale * d) >= kx, "found a lexicographically smaller excess vector"
    print("nucleolus lexicographic minimality on a random 5-player game OK")


def test_gurobi_and_highs_agree():
    """Both back-ends give the same least core / nucleolus / core ranges / min-norm point (small games fit
    the size-limited Gurobi licence; with a full licence this also holds for n = 15)."""
    from carbonfarm.game import backend
    from carbonfarm import make_instance
    from carbonfarm.model.solver import CoalitionSolver
    from carbonfarm.game.coalitions import exact_game, deviation_values
    if not backend.gurobi_available():
        print("gurobi not available: skipped"); return
    inst = make_instance(9, 42)
    sv = CoalitionSolver(inst); gc = sv.solve(range(9))
    A, v = exact_game(sv); vh = deviation_values(A, v, gc.g); p = gc.value - gc.g.sum()
    out = {}
    for name in ("highs", "gurobi"):
        backend.set_backend(name)
        lc = least_core(p, A, vh); nuc = nucleolus(p, A, vh); iv = core_intervals(p, A, vh)
        out[name] = (lc["epsilon_star"], lc["allocation"], nuc["allocation"], iv)
    backend.set_backend("auto")
    a, b = out["highs"], out["gurobi"]
    assert abs(a[0] - b[0]) < 1e-6 and np.allclose(a[1], b[1], atol=1e-3) and np.allclose(a[2], b[2], atol=1e-4)
    assert np.allclose(a[3], b[3], atol=1e-4)
    print("Gurobi == HiGHS on a 9-farmer panel (eps*, min-norm least core, nucleolus, core ranges)")


if __name__ == "__main__":
    test_nucleolus_talmud(); test_shapley_properties(); test_core_two_player_and_centre()
    test_empty_core(); test_nucleolus_is_lexicographically_minimal_on_random_game(); test_gurobi_and_highs_agree()
    print("ALL PASSED")
