"""
Test the clustered_inference module
"""

import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from numpy.testing import assert_almost_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.scenario import multivariate_temporal_simulation
from hidimstat.clustered_inference import clustered_inference


def test_clustered_inference():

    n_samples, n_features = 100, 2000
    support_size = 15
    sigma = 5.0
    rho = 0.95
    n_clusters = 200
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    X_init, y, beta, epsilon = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   rho=rho, shuffle=False, seed=2)

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(n_clusters=n_clusters,
                                connectivity=connectivity,
                                linkage='ward')

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
        clustered_inference(X_init, y, ward, n_clusters)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr[:interior_support],
                        expected[:interior_support])
    assert_almost_equal(pval_corr[extended_support:200],
                        expected[extended_support:200],
                        decimal=1)

    n_samples, n_features, n_times = 100, 2000, 10
    support_size = 15
    sigma = 5.0
    rho_noise = 0.9
    rho_data = 0.95
    n_clusters = 200
    margin_size = 7
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    X, Y, beta, noise = \
        multivariate_temporal_simulation(n_samples=n_samples,
                                         n_features=n_features,
                                         n_times=n_times,
                                         support_size=support_size,
                                         sigma=sigma,
                                         rho_noise=rho_noise,
                                         rho_data=rho_data,
                                         shuffle=False)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(n_clusters=n_clusters,
                                connectivity=connectivity,
                                linkage='ward')

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
        clustered_inference(X, Y, ward, n_clusters,
                            method='desparsified-group-lasso')

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr[:interior_support],
                        expected[:interior_support])
    assert_almost_equal(pval_corr[extended_support:200],
                        expected[extended_support:200],
                        decimal=1)
