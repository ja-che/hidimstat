"""
Test the stat module
"""

import numpy as np
from nose.tools import assert_almost_equal

from hidimstat.stat_tools import sf_from_cb, cdf_from_cb, sf_from_scale


def test_sf_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])
    sf, sf_corr = sf_from_cb(cb_min, cb_max)
    expected = np.asarray([[1.0, 0.025, 0.5], [1.0, 0.075, 0.5]])

    for i in np.arange(expected.shape[1]):
        assert_almost_equal(sf[i], expected[0, i], places=2)
        assert_almost_equal(sf_corr[i], expected[1, i], places=2)


def test_cdf_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])
    sf, sf_corr = cdf_from_cb(cb_min, cb_max)
    expected = np.asarray([[0.0, 0.975, 0.5], [0.0, 0.925, 0.5]])

    for i in np.arange(expected.shape[1]):
        assert_almost_equal(sf[i], expected[0, i], places=2)
        assert_almost_equal(sf_corr[i], expected[1, i], places=2)


def test_sf_from_scale():

    beta = np.asarray([-1.5, 1, 0])
    scale = np.asarray([0.25, 0.5, 0.5])
    sf, sf_corr = sf_from_scale(beta, scale)
    expected = np.asarray([[1.0, 0.022, 0.5], [1.0, 0.066, 0.5]])

    for i in np.arange(expected.shape[1]):
        assert_almost_equal(sf[i], expected[0, i], places=2)
        assert_almost_equal(sf_corr[i], expected[1, i], places=2)
