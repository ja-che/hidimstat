"""
Test the permutation test module
"""

import numpy as np
from numpy.testing import assert_almost_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.permutation_test import permutation_test_cv


def test_permutation_test():

    n_samples, n_features = 20, 50
    support_size = 1
    sigma = 0.1
    rho = 0.0

    X_init, y, beta, epsilon = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   rho=rho, shuffle=False, seed=3)

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    sf_corr, cdf_corr = permutation_test_cv(X_init, y, n_permutations=100)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(sf_corr, expected, decimal=1)