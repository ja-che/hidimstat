"""
Test the desparsified_lasso module
"""

import numpy as np
from numpy.testing import assert_almost_equal

from hidimstat.desparsified_lasso import desparsified_lasso_confint


def test_desparsified_lasso_confint():

    rng = np.random.default_rng(0)

    n_samples, n_features = 30, 30
    n_support = 1
    sigma = 0.02

    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[:n_support] = 1.0
    epsilon = sigma * rng.normal(size=n_samples)
    y = np.dot(X, beta) + epsilon

    beta_hat, cb_min, cb_max = desparsified_lasso_confint(X, y, n_jobs=1)
    expected = np.array([beta, 0.03 * np.ones(n_features),
                         0.06 * np.ones(n_features)])

    for i in np.arange(expected.shape[1]):
        assert_almost_equal(beta_hat[i], expected[0, i], decimal=1)
        assert_almost_equal(beta_hat[i] - cb_min[i], expected[1, i], decimal=1)
        assert_almost_equal(cb_max[i] - cb_min[i], expected[2, i], decimal=1)
