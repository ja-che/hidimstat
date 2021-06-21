"""
Test the stat module
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from hidimstat.stat_tools import \
    (_replace_infinity, pval_corr_from_pval, pval_from_scale,
     zscore_from_cb, pval_from_cb, two_sided_pval_from_zscore,
     two_sided_pval_from_cb, zscore_from_pval,
     zscore_from_one_sided_pvals, pval_from_two_sided_pval_and_sign,
     one_sided_pvals_from_two_sided_pval_and_sign, two_sided_pval_from_pval,
     two_sided_pval_from_one_sided_pvals)


def test__replace_infinity():

    x = np.asarray([10, np.inf, -np.inf])

    # replace inf by the largest absolute value times two
    x_clean = _replace_infinity(x)
    expected = np.asarray([10, 20, -20])
    assert_equal(x_clean, expected)

    # replace inf by 40
    x_clean = _replace_infinity(x, replace_val=40)
    expected = np.asarray([10, 40, -40])
    assert_equal(x_clean, expected)

    # replace inf by the largest absolute value plus one
    x_clean = _replace_infinity(x, method='plus-one')
    expected = np.asarray([10, 11, -11])
    assert_equal(x_clean, expected)


def test_pval_corr_from_pval():

    pval = np.asarray([1.0, 0.025, 0.5])

    # Correction for multiple testing: 3 features tested simultaneously
    pval_corr = pval_corr_from_pval(pval)
    expected = np.asarray([1.0, 0.075, 0.5])
    assert_almost_equal(pval_corr, expected, decimal=10)

    one_minus_pval = np.asarray([0.0, 0.975, 0.5])

    # Correction for multiple testing: 3 features tested simultaneously
    one_minus_pval_corr = pval_corr_from_pval(one_minus_pval)
    expected = np.asarray([0.0, 0.925, 0.5])
    assert_almost_equal(one_minus_pval_corr, expected, decimal=10)


def test_pval_from_scale():

    beta = np.asarray([-1.5, 1, 0])
    scale = np.asarray([0.25, 0.5, 0.5])

    # Testing negativity: low p-value means that we may reject negativity.
    pval, pval_corr = pval_from_scale(beta, scale, testing_sign='minus')
    expected = np.asarray([[1.0, 0.022, 0.5], [1.0, 0.068, 0.5]])

    assert_almost_equal(pval, expected[0], decimal=2)
    assert_almost_equal(pval_corr, expected[1], decimal=2)

    # Testing positivity: low p-value means that we may reject positivity.
    one_minus_pval, one_minus_pval_corr = \
        pval_from_scale(beta, scale, testing_sign='plus')
    expected = np.asarray([[0.0, 0.978, 0.5], [0.0, 0.932, 0.5]])

    assert_almost_equal(one_minus_pval, expected[0], decimal=2)
    assert_almost_equal(one_minus_pval_corr, expected[1], decimal=2)

    # Testing error message: testing_sign must be 'plus' or 'minus'
    np.testing.assert_raises(ValueError, pval_from_scale, beta=beta,
                             scale=scale, testing_sign=None)


def test_zscore_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])

    # Computing z-scores from 95% confidence-intervals assuming Gaussianity
    zscore = zscore_from_cb(cb_min, cb_max)
    expected = np.asarray([-5.87, 1.96, 0])

    assert_almost_equal(zscore, expected, decimal=2)


def test_pval_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])

    # Testing negativity: low p-value means that we may reject negativity.
    pval, pval_corr = pval_from_cb(cb_min, cb_max, testing_sign='minus')
    expected = np.asarray([[1.0, 0.025, 0.5], [1.0, 0.075, 0.5]])

    assert_almost_equal(pval, expected[0], decimal=2)
    assert_almost_equal(pval_corr, expected[1], decimal=2)

    # Testing positivity: low p-value means that we may reject positivity.
    one_minus_pval, one_minus_pval_corr = \
        pval_from_cb(cb_min, cb_max, testing_sign='plus')
    expected = np.asarray([[0.0, 0.975, 0.5], [0.0, 0.925, 0.5]])

    assert_almost_equal(one_minus_pval, expected[0], decimal=2)
    assert_almost_equal(one_minus_pval_corr, expected[1], decimal=2)

    # Testing error message: testing_sign must be 'plus' or 'minus'
    np.testing.assert_raises(ValueError, pval_from_cb, cb_min=cb_min,
                             cb_max=cb_max, testing_sign=None)


def test_two_sided_pval_from_zscore():

    zscore = np.asarray([-5.87, 1.96, 0])

    # Computing two-sided pval from z-scores assuming Gaussianity
    two_sided_pval, two_sided_pval_corr = two_sided_pval_from_zscore(zscore)
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(two_sided_pval, expected[0], decimal=2)
    assert_almost_equal(two_sided_pval_corr, expected[1], decimal=2)


def test_two_sided_pval_from_cb():

    cb_min = np.asarray([-2, 0, -1])
    cb_max = np.asarray([-1, 2, 1])

    # Computing two-sided pval from 95% confidence bounds assuming Gaussianity
    two_sided_pval, two_sided_pval_corr = \
        two_sided_pval_from_cb(cb_min, cb_max)
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(two_sided_pval, expected[0], decimal=2)
    assert_almost_equal(two_sided_pval_corr, expected[1], decimal=2)


def test_zscore_from_pval():

    pval = np.asarray([1.0, 0.025, 0.5])

    # Computing z-scores from p-value testing negativity
    zscore = zscore_from_pval(pval, testing_sign='minus')
    expected = np.asarray([-np.inf, 1.96, 0])

    assert_almost_equal(zscore, expected, decimal=2)

    one_minus_pval = np.asarray([0.0, 0.975, 0.5])

    # Computing z-scores from p-value testing positivity
    zscore = zscore_from_pval(one_minus_pval, testing_sign='plus')
    expected = np.asarray([-np.inf, 1.96, 0])

    assert_almost_equal(zscore, expected, decimal=2)

    # Testing error message: testing_sign must be 'plus' or 'minus'
    np.testing.assert_raises(ValueError, zscore_from_pval,
                             one_sided_pval=pval, testing_sign=None)


def test_zscore_from_one_sided_pvals():

    pval = np.asarray([1.0, 0.025, 0.5])
    one_minus_pval = np.asarray([0.0, 0.975, 0.5])

    # Computing z-scores from p-values testing negativity and positivity
    zscore = zscore_from_one_sided_pvals(pval, one_minus_pval)
    expected = _replace_infinity(np.asarray([-np.inf, 1.96, 0]),
                                 replace_val=40, method='plus-one')

    assert_almost_equal(zscore, expected, decimal=2)


def test_pval_from_two_sided_pval_and_sign():

    two_sided_pval = np.asarray([0.025, 0.05, 0.5])
    parameter_sign = np.asarray([-1.0, 1.0, -1.0])

    # One-sided p-value testing negativity from two-sided p-value and sign.
    pval = pval_from_two_sided_pval_and_sign(two_sided_pval, parameter_sign,
                                             testing_sign='minus')
    expected = np.asarray([0.9875, 0.025, 0.75])

    assert_equal(pval, expected)

    # One-sided p-value testing positivity from two-sided p-value and sign.
    one_minus_pval = \
        pval_from_two_sided_pval_and_sign(two_sided_pval, parameter_sign,
                                          testing_sign='plus')
    expected = np.asarray([0.0125, 0.975, 0.25])

    assert_equal(one_minus_pval, expected)

    # Testing error message: testing_sign must be 'plus' or 'minus'
    np.testing.assert_raises(ValueError, pval_from_two_sided_pval_and_sign,
                             two_sided_pval=two_sided_pval,
                             parameter_sign=parameter_sign,
                             testing_sign=None)


def test_one_sided_pvals_from_two_sided_pval_and_sign():

    two_sided_pval = np.asarray([0.025, 0.05, 0.5])
    parameter_sign = np.asarray([-1.0, 1.0, -1.0])

    # One-sided p-values from two-sided p-value and sign.
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
        one_sided_pvals_from_two_sided_pval_and_sign(two_sided_pval,
                                                     parameter_sign)
    expected = np.asarray([[0.9875, 0.025, 0.75], [0.9625, 0.075, 0.5],
                           [0.0125, 0.975, 0.25], [0.0375, 0.925, 0.5]])

    assert_equal(pval, expected[0])
    assert_almost_equal(pval_corr, expected[1])
    assert_equal(one_minus_pval, expected[2])
    assert_almost_equal(one_minus_pval_corr, expected[3])


def test_two_sided_pval_from_pval():

    pval = np.asarray([1.0, 0.025, 0.5])

    # Two-sided p-value from one-sided p-value testing negativity.
    two_sided_pval, two_sided_pval_corr = \
        two_sided_pval_from_pval(pval, testing_sign='minus')
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(two_sided_pval, expected[0], decimal=2)
    assert_almost_equal(two_sided_pval_corr, expected[1], decimal=2)

    one_minus_pval = np.asarray([0.0, 0.975, 0.5])

    # Two-sided p-value from one-sided p-value testing positivity.
    two_sided_pval, two_sided_pval_corr = \
        two_sided_pval_from_pval(one_minus_pval, testing_sign='plus')
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(two_sided_pval, expected[0], decimal=2)
    assert_almost_equal(two_sided_pval_corr, expected[1], decimal=2)

    # Testing error message: testing_sign must be 'plus' or 'minus'
    np.testing.assert_raises(ValueError, two_sided_pval_from_pval,
                             one_sided_pval=pval, testing_sign=None)


def test_two_sided_pval_from_one_sided_pvals():

    pval = np.asarray([1.0, 0.025, 0.5])
    one_minus_pval = np.asarray([0.0, 0.975, 0.5])

    # Two-sided p-value from p-values testing negativity and positivity.
    two_sided_pval, two_sided_pval_corr = \
        two_sided_pval_from_one_sided_pvals(pval, one_minus_pval)
    expected = np.asarray([[0.0, 0.05, 1.0], [0.0, 0.15, 1.0]])

    assert_almost_equal(two_sided_pval, expected[0], decimal=2)
    assert_almost_equal(two_sided_pval_corr, expected[1], decimal=2)
