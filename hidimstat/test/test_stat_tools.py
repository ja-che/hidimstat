"""
Test the stat module
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from hidimstat.stat_tools import \
    (_replace_infinity, sf_corr_from_sf, cdf_corr_from_cdf, sf_from_scale,
     cdf_from_scale, sf_from_cb, cdf_from_cb, pval_from_cb, zscore_from_cb,
     zscore_from_sf, zscore_from_cdf, zscore_from_sf_and_cdf,
     sf_from_pval_and_sign, cdf_from_pval_and_sign,
     sf_and_cdf_from_pval_and_sign, pval_from_zscore, pval_from_sf,
     pval_from_cdf, pval_from_sf_and_cdf)


def test__replace_infinity():

    x = np.asarray([10, np.inf, -np.inf])

    x_clean = _replace_infinity(x)
    expected = np.asarray([10, 20, -20])
    assert_equal(x_clean, expected)

    x_clean = _replace_infinity(x, replace_val=40)
    expected = np.asarray([10, 40, -40])
    assert_equal(x_clean, expected)

    x_clean = _replace_infinity(x, method='plus-one')
    expected = np.asarray([10, 11, -11])
    assert_equal(x_clean, expected)


def test_sf_corr_from_sf():

    sf = np.asarray([1.0, 0.025, 0.5])
    sf_corr = sf_corr_from_sf(sf)
    expected = sf = np.asarray([1.0, 0.075, 0.5])
    assert_almost_equal(sf_corr, expected, decimal=10)


def test_cdf_corr_from_cdf():

    cdf = np.asarray([0.0, 0.975, 0.5])
    cdf_corr = cdf_corr_from_cdf(cdf)
    expected = np.asarray([0.0, 0.925, 0.5])
    assert_almost_equal(cdf_corr, expected, decimal=10)


def test_sf_from_scale():

    beta = np.asarray([-1.5, 1, 0])
    scale = np.asarray([0.25, 0.5, 0.5])

    sf, sf_corr = sf_from_scale(beta, scale)
    expected = np.asarray([[1.0, 0.022, 0.5], [1.0, 0.068, 0.5]])

    assert_almost_equal(sf, expected[0], decimal=2)
    assert_almost_equal(sf_corr, expected[1], decimal=2)


def test_cdf_from_scale():

    beta = np.asarray([-1.5, 1, 0])
    scale = np.asarray([0.25, 0.5, 0.5])

    cdf, cdf_corr = cdf_from_scale(beta, scale)
    expected = np.asarray([[0.0, 0.978, 0.5], [0.0, 0.932, 0.5]])

    assert_almost_equal(cdf, expected[0], decimal=2)
    assert_almost_equal(cdf_corr, expected[1], decimal=2)


def test_sf_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])

    sf, sf_corr = sf_from_cb(cb_min, cb_max)
    expected = np.asarray([[1.0, 0.025, 0.5], [1.0, 0.075, 0.5]])

    assert_almost_equal(sf, expected[0], decimal=2)
    assert_almost_equal(sf_corr, expected[1], decimal=2)


def test_cdf_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])

    sf, sf_corr = cdf_from_cb(cb_min, cb_max)
    expected = np.asarray([[0.0, 0.975, 0.5], [0.0, 0.925, 0.5]])

    assert_almost_equal(sf, expected[0], decimal=2)
    assert_almost_equal(sf_corr, expected[1], decimal=2)


def test_pval_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])

    pval, pval_corr = pval_from_cb(cb_min, cb_max)
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(pval, expected[0], decimal=2)
    assert_almost_equal(pval_corr, expected[1], decimal=2)


def test_zscore_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])

    zscore = zscore_from_cb(cb_min, cb_max)
    expected = np.asarray([-5.87, 1.96, 0])

    assert_almost_equal(zscore, expected, decimal=2)


def test_zscore_from_sf():

    sf = np.asarray([1.0, 0.025, 0.5])

    zscore = zscore_from_sf(sf)
    expected = np.asarray([-np.inf, 1.96, 0])

    assert_almost_equal(zscore, expected, decimal=2)


def test_zscore_from_cdf():

    cdf = np.asarray([0.0, 0.975, 0.5])

    zscore = zscore_from_cdf(cdf)
    expected = np.asarray([-np.inf, 1.96, 0])

    assert_almost_equal(zscore, expected, decimal=2)


def test_zscore_from_sf_and_cdf():

    sf = np.asarray([1.0, 0.025, 0.5])
    cdf = np.asarray([0.0, 0.975, 0.5])

    zscore = zscore_from_sf_and_cdf(sf, cdf)
    expected = _replace_infinity(np.asarray([-np.inf, 1.96, 0]),
                                 replace_val=40, method='plus-one')

    assert_almost_equal(zscore, expected, decimal=2)


def test_sf_from_pval_and_sign():

    pval = np.asarray([0.025, 0.05, 0.5])
    sign = np.asarray([-1.0, 1.0, -1.0])

    sf = sf_from_pval_and_sign(pval, sign)
    expected = np.asarray([0.9875, 0.025, 0.75])

    assert_equal(sf, expected)


def test_cdf_from_pval_and_sign():

    pval = np.asarray([0.025, 0.05, 0.5])
    sign = np.asarray([-1.0, 1.0, -1.0])

    cdf = cdf_from_pval_and_sign(pval, sign)
    expected = np.asarray([0.0125, 0.975, 0.25])

    assert_equal(cdf, expected)


def test_sf_and_cdf_from_pval_and_sign():

    pval = np.asarray([0.025, 0.05, 0.5])
    sign = np.asarray([-1.0, 1.0, -1.0])

    sf, sf_corr, cdf, cdf_corr = sf_and_cdf_from_pval_and_sign(pval, sign)
    expected = np.asarray([[0.9875, 0.025, 0.75], [0.9625, 0.075, 0.5],
                           [0.0125, 0.975, 0.25], [0.0375, 0.925, 0.5]])

    assert_equal(sf, expected[0])
    assert_almost_equal(sf_corr, expected[1])
    assert_equal(cdf, expected[2])
    assert_almost_equal(cdf_corr, expected[3])


def test_pval_from_zscore():

    zscore = np.asarray([-5.87, 1.96, 0])

    pval, pval_corr = pval_from_zscore(zscore)
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(pval, expected[0], decimal=2)
    assert_almost_equal(pval_corr, expected[1], decimal=2)


def test_pval_from_sf():

    sf = np.asarray([1.0, 0.025, 0.5])

    pval, pval_corr = pval_from_sf(sf)
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(pval, expected[0], decimal=2)
    assert_almost_equal(pval_corr, expected[1], decimal=2)


def test_pval_from_cdf():

    cdf = np.asarray([0.0, 0.975, 0.5])

    pval, pval_corr = pval_from_cdf(cdf)
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(pval, expected[0], decimal=2)
    assert_almost_equal(pval_corr, expected[1], decimal=2)


def test_pval_from_sf_and_cdf():

    sf = np.asarray([1.0, 0.025, 0.5])
    cdf = np.asarray([0.0, 0.975, 0.5])

    pval, pval_corr = pval_from_sf_and_cdf(sf, cdf)
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(pval, expected[0], decimal=2)
    assert_almost_equal(pval_corr, expected[1], decimal=2)
