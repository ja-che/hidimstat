import numpy as np
from numpy.testing import assert_array_almost_equal

from hidimstat.data_simulation import simu_data

n = 100
p = 200
seed = 42


def test_simu_data():
    X, y, _, _ = simu_data(n, p, seed=seed)

    assert X.shape == (n, p)
    assert y.size == n
