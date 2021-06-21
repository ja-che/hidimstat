"""
Test the ensemble_clustered_inference module
"""

import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from numpy.testing import assert_almost_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.ensemble_clustered_inference import ensemble_clustered_inference


def test_ensemble_clustered_inference():

    n_samples, n_features = 100, 2000
    support_size = 15
    sigma = 5.0
    rho = 0.95

    X_init, y, beta, epsilon = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   rho=rho, shuffle=False, seed=0)

    margin_size = 5
    n_clusters = 200
    n_bootstraps = 3

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(n_clusters=n_clusters,
                                connectivity=connectivity,
                                linkage='ward')

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
        ensemble_clustered_inference(X_init, y, ward, n_clusters,
                                     n_bootstraps=n_bootstraps)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr[:support_size-margin_size],
                        expected[:support_size-margin_size])
    assert_almost_equal(pval_corr[support_size+margin_size:200],
                        expected[support_size+margin_size:200],
                        decimal=1)
