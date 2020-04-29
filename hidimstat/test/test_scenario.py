"""
Test the scenario module
"""

import numpy as np
from nose.tools import assert_almost_equal, assert_equal

from hidimstat.scenario import design_matrix_toeplitz_cov, scenario


def test_design_matrix_toeplitz_cov():

    rho = 0.7

    X = design_matrix_toeplitz_cov(rho=rho)
    n_samples, n_features = X.shape

    rho_hat = 0

    for i in np.arange(n_features - 1):
        rho_hat += np.corrcoef(X[:, i], X[:, i + 1])[0, 1] / (n_features - 1)

    assert_almost_equal(rho_hat, rho, places=1)


def test_scenario():

    effect_s_nb = 1
    effect_m_nb = 2
    effect_l_nb = 3

    y, beta, X, epsilon = scenario(effect_s_nb=effect_s_nb,
                                   effect_m_nb=effect_m_nb,
                                   effect_l_nb=effect_l_nb)
    y_hat = np.dot(X, beta) + epsilon

    n_samples, n_features = X.shape

    assert_equal(beta[beta == 0.25].size, effect_s_nb)
    assert_equal(beta[beta == 0.5].size, effect_m_nb)
    assert_equal(beta[beta == 1].size, effect_l_nb)

    for i in np.arange(n_samples):
        assert_equal(y_hat[i], y[i])
