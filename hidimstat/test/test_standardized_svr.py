"""
Test the standardized_svr module
"""

import numpy as np
from numpy.testing import assert_almost_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.stat_tools import sf_from_scale
from hidimstat.standardized_svr import standardized_svr


def test_standardized_svr():

    n_samples, n_features = 100, 2000
    support_size = 15
    sigma = 5.0
    rho = 0.9
    margin_size = 5

    X_init, y, beta, noise = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   rho=rho, shuffle=False, seed=0)

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    beta_hat, scale_hat = standardized_svr(X_init, y)

    sf, sf_corr = sf_from_scale(beta_hat, scale_hat)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    assert_almost_equal(sf_corr[:interior_support],
                        expected[:interior_support],
                        decimal=1)
    assert_almost_equal(sf_corr[extended_support:200],
                        expected[extended_support:200],
                        decimal=1)
