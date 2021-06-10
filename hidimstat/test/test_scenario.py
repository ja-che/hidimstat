"""
Test the scenario module
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.scenario import multivariate_simulation
from hidimstat.scenario import multivariate_temporal_simulation

ROI_SIZE_2D = 2
SHAPE_2D = (12, 12)

ROI_SIZE_3D = 2
SHAPE_3D = (12, 12, 12)


def test_multivariate_1D_simulation():

    n_samples = 100
    n_features = 500
    support_size = 10
    rho = 0.7
    sigma = 1.0

    X, y, beta, epsilon = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   rho=rho, shuffle=False, seed=0)

    sigma_hat = np.std(epsilon)
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]

    assert_almost_equal(sigma_hat, sigma, decimal=1)
    assert_almost_equal(rho_hat, rho, decimal=1)
    assert_equal(X.shape, (n_samples, n_features))
    assert_equal(np.count_nonzero(beta), support_size)
    assert_equal(y, np.dot(X, beta) + epsilon)

    X, y, beta, epsilon = \
        multivariate_1D_simulation()
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]
    assert_almost_equal(rho_hat, 0, decimal=1)


def test_multivariate_simulation():

    n_samples = 100
    shape = SHAPE_2D
    roi_size = ROI_SIZE_2D
    sigma = 1.0
    smooth_X = 1.0
    rho_expected = 0.8
    return_shaped_data = True

    X, y, beta, epsilon, X_, w = \
        multivariate_simulation(n_samples=n_samples, shape=shape,
                                roi_size=roi_size, sigma=sigma,
                                smooth_X=smooth_X,
                                return_shaped_data=return_shaped_data,
                                seed=0)

    sigma_hat = np.std(epsilon)
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]

    assert_almost_equal(sigma_hat, sigma, decimal=1)
    assert_almost_equal(rho_hat, rho_expected, decimal=2)
    assert_equal(X.shape, (n_samples, shape[0] * shape[1]))
    assert_equal(X_.shape, (n_samples, shape[0], shape[1]))
    assert_equal(np.count_nonzero(beta), 4 * (roi_size ** 2))
    assert_equal(y, np.dot(X, beta) + epsilon)

    shape = SHAPE_3D
    roi_size = ROI_SIZE_3D
    return_shaped_data = False

    X, y, beta, epsilon = \
        multivariate_simulation(n_samples=n_samples, shape=shape,
                                roi_size=roi_size,
                                return_shaped_data=return_shaped_data,
                                seed=0)

    assert_equal(X.shape, (n_samples, shape[0] * shape[1] * shape[2]))
    assert_equal(np.count_nonzero(beta), 5 * (roi_size ** 3))


def test_multivariate_temporal_simulation():

    n_samples = 30
    n_features = 50
    n_targets = 10
    support_size = 2
    sigma = 1.0
    rho = 0.9

    X, Y, Beta, E = \
        multivariate_temporal_simulation(n_samples=n_samples,
                                         n_features=n_features,
                                         n_targets=n_targets,
                                         support_size=support_size,
                                         sigma=sigma, rho=rho)

    sigma_hat = np.std(E[:, -1])
    rho_hat = np.corrcoef(E[:, -1], E[:, -2])[0, 1]

    assert_almost_equal(sigma_hat, sigma, decimal=1)
    assert_almost_equal(rho_hat, rho, decimal=1)
    assert_equal(X.shape, (n_samples, n_features))
    assert_equal(Y.shape, (n_samples, n_targets))
    assert_equal(np.count_nonzero(Beta), support_size * n_targets)
    assert_equal(Y, np.dot(X, Beta) + E)
