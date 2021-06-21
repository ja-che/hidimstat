"""
Test the desparsified_lasso module
"""

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.linalg import toeplitz

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.scenario import multivariate_temporal_simulation
from hidimstat.desparsified_lasso import desparsified_lasso
from hidimstat.desparsified_lasso import desparsified_group_lasso


def test_desparsified_lasso():

    n_samples, n_features = 20, 50
    support_size = 1
    sigma = 0.1
    rho = 0.0

    X, y, beta, noise = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   rho=rho, shuffle=False, seed=5)

    beta_hat, cb_min, cb_max = desparsified_lasso(X, y)

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_almost_equal(cb_min, beta - 0.05, decimal=1)
    assert_almost_equal(cb_max,  beta + 0.05, decimal=1)

    beta_hat, cb_min, cb_max = desparsified_lasso(X, y, dof_ajdustement=True)

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_almost_equal(cb_min, beta - 0.05, decimal=1)
    assert_almost_equal(cb_max,  beta + 0.05, decimal=1)


def test_desparsified_group_lasso():

    n_samples = 50
    n_features = 100
    n_times = 10
    support_size = 2
    sigma = 0.1
    rho = 0.9
    corr = toeplitz(np.geomspace(1, rho ** (n_times - 1), n_times))
    cov = np.outer(sigma, sigma) * corr

    X, Y, beta, noise = \
        multivariate_temporal_simulation(n_samples=n_samples,
                                         n_features=n_features,
                                         n_times=n_times,
                                         support_size=support_size,
                                         sigma=sigma, rho_noise=rho)

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
        desparsified_group_lasso(X, Y, cov=cov)

    expected_pval_corr = \
        np.concatenate((np.zeros(support_size),
                        0.5 * np.ones(n_features - support_size)))

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_almost_equal(pval_corr, expected_pval_corr, decimal=1)

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
        desparsified_group_lasso(X, Y, test='F')

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_almost_equal(pval_corr, expected_pval_corr, decimal=1)

    bad_cov = np.delete(cov, 0, axis=1)
    np.testing.assert_raises(ValueError, desparsified_group_lasso,
                             X=X, Y=Y, cov=bad_cov)
