"""
Test the noise_std module
"""

import numpy as np
from nose.tools import assert_almost_equal

from noise_std import reid


def test_reid():

    np.random.seed(0)

    n_samples, n_features = 30, 30
    n_support = 10
    sigma = 2.0

    X = np.random.randn(n_samples, n_features)
    beta = np.zeros(n_features)
    beta[:n_support] = 1.0
    epsilon = sigma * np.random.randn(n_samples)
    y = np.dot(X, beta) + epsilon

    sigma_hat = reid(X, y)
    expected = sigma

    assert_almost_equal(sigma_hat / expected, 1.0, places=0)

    sigma_hat = reid(X, y, method='lasso', max_iter=1)
    expected = sigma

    assert_almost_equal(sigma_hat / expected, 1.0, places=0)
