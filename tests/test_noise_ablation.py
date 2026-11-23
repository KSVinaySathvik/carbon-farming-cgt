"""Changing cmp_noise must regenerate the same panel and keep CMP_ij positive."""
import numpy as np

from carbonfarm import make_instance
from carbonfarm.config import Params
from experiments.ablations.a7_cmp_noise import GRID, apply


def test_same_panel_rescaled_noise():
    base = make_instance(15, 42)
    for v in GRID:
        new = apply(base, v)
        assert np.array_equal(new.farm_size, base.farm_size)
        assert np.array_equal(new.budget_noise, base.budget_noise)
        ratio = new.cmp / base.practices.cmp_base[None, :]
        assert ratio.min() >= 1 - v - 1e-12 and ratio.max() <= 1 + v + 1e-12
        assert (new.cmp > 0).all()
    assert apply(base, Params().cmp_noise).cmp.shape == base.cmp.shape
    assert np.allclose(apply(base, Params().cmp_noise).cmp, base.cmp)


if __name__ == "__main__":
    test_same_panel_rescaled_noise(); print("ok")
