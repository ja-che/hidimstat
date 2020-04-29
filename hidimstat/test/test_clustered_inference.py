"""
Test the clustered_inference module
"""

import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from nose.tools import assert_almost_equal

from hidimstat.scenario import scenario

from hidimstat.clustered_inference import clustered_inference


def test_clustered_inference():

    scenario_type = 'Toeplitz'
    seed = 0
    n_samples, n_features = 100, 2000
    effect_small, effect_medium, effect_large = 0.25, 0.5, 1.0
    effect_s_nb, effect_m_nb, effect_l_nb = 0, 0, 17
    sigma = 5.0
    rho = 0.95
    shuffle = False

    n_clusters = 200
    n_support = effect_s_nb + effect_m_nb + effect_l_nb

    y, beta, X_init, epsilon = scenario(
        scenario_type, seed, n_samples, n_features, effect_small,
        effect_medium, effect_large, effect_s_nb, effect_m_nb, effect_l_nb,
        sigma, rho, shuffle)

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(n_clusters=n_clusters,
                                connectivity=connectivity,
                                linkage='ward')

    sf, sf_corr, cdf, cdf_corr = \
        clustered_inference(X_init, y, ward, n_clusters, method='DL')

    expected = 0.5 * np.ones(n_features)
    expected[:n_support] = 0.0

    for i in np.arange(expected.size):
        assert_almost_equal(sf_corr[i], expected[i], places=2)
