"""
Test the desparsified_lasso module
"""

from numpy.testing import assert_almost_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.desparsified_lasso import desparsified_lasso_confint


def test_desparsified_lasso_confint():

    n_samples, n_features = 20, 50
    support_size = 1
    sigma = 0.1
    rho = 0.0

    X, y, beta, noise = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   rho=rho, shuffle=False, seed=5)

    beta_hat, cb_min, cb_max = desparsified_lasso_confint(X, y)

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_almost_equal(cb_min, beta - 0.05, decimal=1)
    assert_almost_equal(cb_max,  beta + 0.05, decimal=1)

    beta_hat, cb_min, cb_max = \
        desparsified_lasso_confint(X, y, normalize=True, dof_ajdustement=True)

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_almost_equal(cb_min, beta - 0.05, decimal=1)
    assert_almost_equal(cb_max,  beta + 0.05, decimal=1)
