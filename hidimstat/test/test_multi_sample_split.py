"""
Test the multi_sample_split module
"""

import numpy as np
from nose.tools import assert_almost_equal

from hidimstat.multi_sample_split import aggregate_medians, aggregate_quantiles


def test_aggregate_medians():

    n_iter, n_features = 20, 5
    list_sf = (1.0 / (np.arange(n_iter * n_features) + 1))
    list_sf = list_sf.reshape((n_iter, n_features))
    list_sf[15:, :] = 3e-3

    sf = aggregate_medians(list_sf)
    expected = 0.04 * np.ones(n_features)

    for i in np.arange(expected.size):
        assert_almost_equal(sf[i], expected[i], places=2)


def test_aggregate_quantiles():

    n_iter, n_features = 20, 5
    list_sf = (1.0 / (np.arange(n_iter * n_features) + 1))
    list_sf = list_sf.reshape((n_iter, n_features))
    list_sf[15:, :] = 3e-3

    sf = aggregate_quantiles(list_sf)
    expected = 0.04 * np.ones(n_features)

    for i in np.arange(expected.size):
        assert_almost_equal(sf[i], expected[i], places=2)
