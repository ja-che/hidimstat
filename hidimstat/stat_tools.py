import numpy as np
from scipy.stats import norm


def _replace_infinity(x, replace_val=None, method='times-two'):
    """Replace infinity by large value"""

    largest_non_inf = np.max(np.abs(x)[np.abs(x) != np.inf])

    if method == 'times-two':
        replace_val_min = largest_non_inf * 2
    elif method == 'plus-one':
        replace_val_min = largest_non_inf + 1

    if (replace_val is not None) and (replace_val < largest_non_inf):
        replace_val = replace_val_min
    elif replace_val is None:
        replace_val = replace_val_min

    x_new = np.copy(x)
    x_new[x_new == np.inf] = replace_val
    x_new[x_new == -np.inf] = -replace_val

    return x_new


def pval_corr_from_pval(one_sided_pval):
    """Computing one-sided p-values corrrected for multiple testing
    from simple testing one-sided p-values.

    Parameters
    ----------
    one_sided_pval : ndarray, shape (n_features,)
        One-sided p-values.

    Returns
    -------
    one_sided_pval_corr : ndarray, shape (n_features,)
        Corrected one-sided p-values.
     """

    n_features = one_sided_pval.size

    one_sided_pval_corr = np.zeros(n_features) + 0.5

    ind = (one_sided_pval < 0.5)
    one_sided_pval_corr[ind] = \
        np.minimum(one_sided_pval[ind] * n_features, 0.5)

    ind = (one_sided_pval > 0.5)
    one_sided_pval_corr[ind] = \
        np.maximum(1 - (1 - one_sided_pval[ind]) * n_features, 0.5)

    return one_sided_pval_corr


def pval_from_scale(beta, scale, testing_sign, eps=1e-14):
    """Computing one-sided p-values from the value of the parameter
    and its scale.

    Parameters
    ----------
    beta : ndarray, shape (n_features,)
        Value of the parameters.

    scale : ndarray, shape (n_features,)
        Value of the standard deviation of the parameters.

    testing_sign : {'plus', 'minus'}
        Sign tested in the alternative hypothesis of the statistical tests
        to compute the one-sided p-values (`one_sided_pval`).
        If 'plus', low p-values characterize positive effect sizes.
        If 'minus', low p-values characterize negative effect sizes.

    eps : float, optional
        Machine-precision regularization in the computation of the p-values.

    Returns
    -------
    one_sided_pval : ndarray, shape (n_features,)
        One-sided p-values (test the null, alternative hypothesis is signed).

    one_sided_pval_corr : ndarray, shape (n_features,)
        Corrected one-sided p-values.
    """

    n_features = beta.size

    index_no_nan = tuple([scale != 0.0])

    one_sided_pval = np.zeros(n_features) + 0.5

    if testing_sign == 'plus':
        one_sided_pval[index_no_nan] = \
            norm.sf(beta[index_no_nan], scale=scale[index_no_nan])
    elif testing_sign == 'minus':
        one_sided_pval[index_no_nan] = \
            norm.cdf(beta[index_no_nan], scale=scale[index_no_nan])
    else:
        raise ValueError("testing_sign must be either 'minus' or 'plus'.")

    one_sided_pval[one_sided_pval > 1 - eps] = 1 - eps
    one_sided_pval_corr = pval_corr_from_pval(one_sided_pval)

    return one_sided_pval, one_sided_pval_corr


def zscore_from_cb(cb_min, cb_max, confidence=0.95, distrib='norm'):
    """Computing z-scores from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound.

    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound.

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-scores.
    """

    if distrib == 'norm':
        quantile = norm.ppf(1 - (1 - confidence) / 2)

    beta_hat = (cb_min + cb_max) / 2

    zscore = beta_hat / (cb_max - cb_min) * 2 * quantile

    return zscore


def pval_from_cb(cb_min, cb_max, testing_sign, confidence=0.95,
                 distrib='norm', eps=1e-14):
    """Computing one-sided p-values from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound.

    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound.

    testing_sign : {'plus', 'minus'}
        Sign tested in the alternative hypothesis of the statistical tests
        to compute the one-sided p-values (`one_sided_pval`).
        If 'plus', low p-values characterize positive effect sizes.
        If 'minus', low p-values characterize negative effect sizes.

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    eps : float, optional
        Machine-precision regularization in the computation of the p-values.

    Returns
    -------
    one_sided_pval : ndarray, shape (n_features,)
        One-sided p-values (test the null, alternative hypothesis is signed).

    one_sided_pval_corr : ndarray, shape (n_features,)
        Corrected one-sided p-values.
    """

    zscore = \
        zscore_from_cb(cb_min, cb_max, confidence=confidence, distrib=distrib)

    if testing_sign == 'plus':

        if distrib == 'norm':
            one_sided_pval = norm.sf(zscore)

    elif testing_sign == 'minus':

        if distrib == 'norm':
            one_sided_pval = norm.cdf(zscore)

    else:

        raise ValueError("testing_sign must be either 'minus' or 'plus'.")

    one_sided_pval[one_sided_pval > 1 - eps] = 1 - eps
    one_sided_pval_corr = pval_corr_from_pval(one_sided_pval)

    return one_sided_pval, one_sided_pval_corr


def two_sided_pval_from_zscore(zscore, distrib='norm'):
    """Computing two-sided p-values from z-scores.

    Parameters
    ----------
    zscore : ndarray, shape (n_features,)
        z-scores.

    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    two_sided_pval_corr : ndarray, shape (n_features,)
        Two-sided p-values corrected for multiple testing.
    """
    n_features = zscore.size

    if distrib == 'norm':
        two_sided_pval = 2 * norm.sf(np.abs(zscore))

    two_sided_pval_corr = np.minimum(1, two_sided_pval * n_features)

    return two_sided_pval, two_sided_pval_corr


def two_sided_pval_from_cb(cb_min, cb_max, confidence=0.95, distrib='norm'):
    """Computing two-sided p-values from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound.

    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound.

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    two_sided_pval_corr : ndarray, shape (n_features,)
        Two-sided p-values corrected for multiple testing.
    """
    zscore = \
        zscore_from_cb(cb_min, cb_max, confidence=confidence, distrib=distrib)

    two_sided_pval, two_sided_pval_corr = \
        two_sided_pval_from_zscore(zscore, distrib='norm')

    return two_sided_pval, two_sided_pval_corr


def zscore_from_pval(one_sided_pval, testing_sign, distrib='norm'):
    """Computing z-scores from one-sided p-values.

    Parameters
    -----------
    one_sided_pval : ndarray, shape (n_features,)
        One-sided p-values (test the null, alternative hypothesis is signed).

    testing_sign : {'plus', 'minus'}
        Sign tested in the alternative hypothesis of the statistical tests
        to compute the one-sided p-values (`one_sided_pval`).
        If 'plus', low p-values characterize positive effect sizes.
        If 'minus', low p-values characterize negative effect sizes.

    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-scores.
    """

    if testing_sign == 'plus':

        if distrib == 'norm':
            zscore = norm.isf(one_sided_pval)

    elif testing_sign == 'minus':

        if distrib == 'norm':
            zscore = norm.ppf(one_sided_pval)

    else:

        raise ValueError("testing_sign must be either 'minus' or 'plus'.")

    return zscore


def zscore_from_one_sided_pvals(pval, one_minus_pval, distrib='norm'):
    """Computing z-scores from one-sided p-values testing for positivity
    and from one-sided p-values testing for negativity.

    Parameters
    -----------
    pval : ndarray, shape (n_features,)
        One-sided p-values, low p-values characterize positive effect sizes.

    one_minus_pval : ndarray, shape (n_features,)
        One-sided p-values, low p-values characterize negative effect sizes.

    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-scores.
    """
    ind = (pval < 0.5)

    zscore = \
        zscore_from_pval(one_minus_pval, testing_sign='minus', distrib=distrib)

    zscore[ind] = \
        zscore_from_pval(pval, testing_sign='plus', distrib=distrib)[ind]

    zscore = _replace_infinity(zscore, replace_val=40, method='plus-one')

    return zscore


def pval_from_two_sided_pval_and_sign(two_sided_pval, parameter_sign,
                                      testing_sign, eps=1e-14):
    """Computing one-sided p-values (for testing positivity or negativity)
    from p-value and parameter sign.

    Parameters
    ----------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    parameter_sign : ndarray, shape (n_features,)
        Estimated signs for the parameters.

    testing_sign : {'plus', 'minus'}
        Sign tested in the alternative hypothesis of the statistical tests
        to compute the one-sided p-values (`one_sided_pval`).
        If 'plus', low p-values characterize positive effect sizes.
        If 'minus', low p-values characterize negative effect sizes.

    eps : float, optional
        Machine-precision regularization in the computation of the p-values.

    Returns
    -------
    one_sided_pval : ndarray, shape (n_features,)
        One-sided p-values (test the null, alternative hypothesis is signed).
    """

    n_features = two_sided_pval.size
    one_sided_pval = 0.5 * np.ones(n_features)

    if testing_sign == 'plus':

        one_sided_pval[parameter_sign > 0] = \
            two_sided_pval[parameter_sign > 0] / 2

        one_sided_pval[parameter_sign < 0] = \
            1 - two_sided_pval[parameter_sign < 0] / 2

    elif testing_sign == 'minus':

        one_sided_pval[parameter_sign > 0] = \
            1 - two_sided_pval[parameter_sign > 0] / 2

        one_sided_pval[parameter_sign < 0] = \
            two_sided_pval[parameter_sign < 0] / 2

    else:

        raise ValueError("testing_sign must be either 'minus' or 'plus'.")

    one_sided_pval[one_sided_pval > 1 - eps] = 1 - eps

    return one_sided_pval


def one_sided_pvals_from_two_sided_pval_and_sign(two_sided_pval,
                                                 parameter_sign, eps=1e-14):
    """Computing both one-sided p-values for testing both negativity and
    positivity from two-sided p-values and parameter sign.

    Parameters
    ----------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    parameter_sign : ndarray, shape (n_features,)
        Estimated signs for the parameters.

    eps : float, optional
        Machine-precision regularization in the computation of the p-values.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        One-sided p-values, low p-values characterize positive effect sizes.

    pval_corr : ndarray, shape (n_features,)
        Corrected p-values, low p-values characterize positive effect sizes.

    one_minus_pval : ndarray, shape (n_features,)
        One-sided p-values, low p-values characterize negative effect sizes.

    one_minus_pval_corr : ndarray, shape (n_features,)
        Corrected p-values, low p-values characterize negative effect sizes.
    """

    pval = pval_from_two_sided_pval_and_sign(two_sided_pval, parameter_sign,
                                             testing_sign='plus', eps=eps)
    one_minus_pval = \
        pval_from_two_sided_pval_and_sign(two_sided_pval, parameter_sign,
                                          testing_sign='minus', eps=eps)

    pval_corr = pval_corr_from_pval(pval)
    one_minus_pval_corr = pval_corr_from_pval(one_minus_pval)

    return pval, pval_corr, one_minus_pval, one_minus_pval_corr


def two_sided_pval_from_pval(one_sided_pval, testing_sign, distrib='norm'):
    """Computing two-sided p-value from one-sided p-values testing
    for negativity or positivity.

    Parameters
    ----------
    one_sided_pval : ndarray, shape (n_features,)
        One-sided p-values (test the null, alternative hypothesis is signed).

    testing_sign : {'plus', 'minus'}
        Sign tested in the alternative hypothesis of the statistical tests
        to compute the one-sided p-values (`one_sided_pval`).
        If 'plus', low p-values characterize positive effect sizes.
        If 'minus', low p-values characterize negative effect sizes.

    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    two_sided_pval_corr : ndarray, shape (n_features,)
        Two-sided p-values corrected for multiple testing.
    """

    zscore = \
        zscore_from_pval(one_sided_pval, testing_sign=testing_sign,
                         distrib=distrib)

    two_sided_pval, two_sided_pval_corr = \
        two_sided_pval_from_zscore(zscore, distrib=distrib)

    return two_sided_pval, two_sided_pval_corr


def two_sided_pval_from_one_sided_pvals(pval, one_minus_pval, distrib='norm'):
    """Computing two-sided p-value from both one-sided p-values testing
    for both negativity and positivity.

    Parameters
    ----------
    pval : ndarray, shape (n_features,)
        One-sided p-values, low p-values characterize positive effect sizes.

    one_minus_pval : ndarray, shape (n_features,)
        One-sided p-values, low p-values characterize negative effect sizes.

    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    two_sided_pval_corr : ndarray, shape (n_features,)
        Two-sided p-values corrected for multiple testing.
    """

    zscore = \
        zscore_from_one_sided_pvals(pval, one_minus_pval, distrib='norm')

    two_sided_pval, two_sided_pval_corr = \
        two_sided_pval_from_zscore(zscore, distrib=distrib)

    return two_sided_pval, two_sided_pval_corr
