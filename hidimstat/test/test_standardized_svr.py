"""
Test the standardized_svr module
"""

import numpy as np
from nose.tools import assert_almost_equal

from hidimstat.scenario import scenario
from hidimstat.stat_tools import sf_from_scale
from hidimstat.standardized_svr import standardized_svr


def test_clustered_inference():

    scenario_type = 'Toeplitz'
    seed = 0
    n_samples, n_features = 100, 2000
    effect_small, effect_medium, effect_large = 0.25, 0.5, 1.0
    effect_s_nb, effect_m_nb, effect_l_nb = 0, 0, 17
    sigma = 5.0
    rho = 0.9
    shuffle = False

    n_support = effect_s_nb + effect_m_nb + effect_l_nb

    y, beta, X_init, epsilon = scenario(
        scenario_type, seed, n_samples, n_features, effect_small,
        effect_medium, effect_large, effect_s_nb, effect_m_nb, effect_l_nb,
        sigma, rho, shuffle)

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    beta_hat, scale_hat = standardized_svr(X_init, y)

    sf, sf_corr = sf_from_scale(beta_hat, scale_hat)

    sf_corr[15:20]

    expected = 0.5 * np.ones(n_features)
    expected[:n_support] = 0.0

    for i in np.arange(expected.size):
        assert_almost_equal(sf_corr[i], expected[i], places=1)
