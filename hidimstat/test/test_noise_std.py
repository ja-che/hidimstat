"""
Test the noise_std module
"""

from numpy.testing import assert_almost_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.noise_std import reid, empirical_snr


def test_reid():

    n_samples, n_features = 30, 30
    sigma = 2.0

    # First test
    # ##########
    support_size = 10

    X, y, beta, epsilon = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   seed=0)

    # max_iter=1 to get a better coverage
    sigma_hat, _ = reid(X, y, tol=1e-3, max_iter=1)
    expected = sigma

    assert_almost_equal(sigma_hat / expected, 1.0, decimal=0)

    # Second test
    # ###########
    support_size = 0

    X, y, beta, epsilon = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   seed=1)

    sigma_hat, _ = reid(X, y)
    expected = sigma

    assert_almost_equal(sigma_hat / expected, 1.0, decimal=0)


def test_empirical_snr():

    n_samples, n_features = 30, 30
    support_size = 10
    sigma = 2.0

    X, y, beta, epsilon = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   seed=0)

    snr = empirical_snr(X, y, beta)
    expected = 2.0

    assert_almost_equal(snr / expected, 1.0, decimal=0)
