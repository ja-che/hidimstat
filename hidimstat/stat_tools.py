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


def sf_corr_from_sf(sf):
    """Computing survival function values corrrected for multiple testing
    from simple testing survival function values.

    Parameters
    ----------
    sf : ndarray, shape (n_features,)
        Survival function values

    Returns
    -------
    sf_corr : ndarray, shape (n_features,)
        Corrected survival function values
     """

    n_features = sf.size

    sf_corr = np.zeros(n_features) + 0.5

    sf_corr[sf < 0.5] = np.minimum(0.5, sf[sf < 0.5] * n_features)
    sf_corr[sf > 0.5] = np.maximum(0.5, 1 - (1 - sf[sf > 0.5]) * n_features)

    return sf_corr


def cdf_corr_from_cdf(cdf):
    """Computing cumulative distribution function values corrrected for
    multiple testing from simple testing cumulative distribution function
    values.

    Parameters
    ----------
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values

    Returns
    -------
    cdf_corr : ndarray, shape (n_features,)
        Corrected cumulative distribution function values
     """

    n_features = cdf.size

    cdf_corr = np.zeros(n_features) + 0.5

    cdf_corr[cdf < 0.5] = np.minimum(0.5, cdf[cdf < 0.5] * n_features)
    cdf_corr[cdf > 0.5] = \
        np.maximum(0.5, 1 - (1 - cdf[cdf > 0.5]) * n_features)

    return cdf_corr


def sf_from_scale(beta, scale, eps=1e-14):
    """Survival function values from the value of the parameter and its scale.

    Parameters
    ----------
    beta : ndarray, shape (n_features,)
        Value of the parameters
    scale : ndarray, shape (n_features,)
        Value of the variance of the parameters
    eps : float, optional
        The machine-precision regularization in the computation of the
        survival function value

    Returns
    -------
    sf : ndarray, shape (n_features,)
        Survival function values
    sf_corr : ndarray, shape (n_features,)
        Corrected survival function values
    """

    n_features = beta.size

    index_no_nan = tuple([scale != 0.0])

    sf = np.zeros(n_features) + 0.5
    sf[index_no_nan] = norm.sf(beta[index_no_nan], scale=scale[index_no_nan])
    sf[sf > 1 - eps] = 1 - eps
    sf_corr = sf_corr_from_sf(sf)

    return sf, sf_corr


def cdf_from_scale(beta, scale, eps=1e-14):
    """Cumulative distribution function values from the value of the parameter
    and its scale.

    Parameters
    ----------
    beta : ndarray, shape (n_features,)
        Value of the parameters
    scale : ndarray, shape (n_features,)
        Value of the variance of the parameters
    eps : float, optional
        The machine-precision regularization in the computation of the
        cumulative distribution function value

    Returns
    -------
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values
    cdf_corr : ndarray, shape (n_features,)
        Corrected cumulative distribution function values
    """

    n_features = beta.size

    index_no_nan = tuple([scale != 0.0])

    cdf = np.zeros(n_features) + 0.5
    cdf[index_no_nan] = norm.cdf(beta[index_no_nan],
                                 scale=scale[index_no_nan])
    cdf[cdf > 1 - eps] = 1 - eps
    cdf_corr = cdf_corr_from_cdf(cdf)

    return cdf, cdf_corr


def sf_from_cb(cb_min, cb_max, confidence=0.95, distrib='norm', eps=1e-14):
    """Survival function values from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound
    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound
    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.
    eps : float, optional
        The machine-precision regularization in the computation of the
        survival function value

    Returns
    -------
    sf : ndarray, shape (n_features,)
        Survival function values
    sf_corr : ndarray, shape (n_features,)
        Corrected survival function values
    """

    if distrib == 'norm':
        quantile = norm.ppf(1 - (1 - confidence) / 2)

    beta_hat = (cb_min + cb_max) / 2

    zscore = beta_hat / (cb_max - cb_min) * 2 * quantile

    if distrib == 'norm':
        sf = norm.sf(zscore)

    sf[sf > 1 - eps] = 1 - eps
    sf_corr = sf_corr_from_sf(sf)

    return sf, sf_corr


def cdf_from_cb(cb_min, cb_max, confidence=0.95, distrib='norm', eps=1e-14):
    """Cumulative function values from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound
    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound
    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.
    eps : float, optional
        The machine-precision regularization in the computation of the
        cumulative distribution function value

    Returns
    -------
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values
    cdf_corr : ndarray, shape (n_features,)
        Corrected cumulative distribution function values
    """

    if distrib == 'norm':
        quantile = norm.ppf(1 - (1 - confidence) / 2)

    beta_hat = (cb_min + cb_max) / 2

    zscore = beta_hat / (cb_max - cb_min) * 2 * quantile

    if distrib == 'norm':
        cdf = norm.cdf(zscore)

    cdf[cdf > 1 - eps] = 1 - eps
    cdf_corr = cdf_corr_from_cdf(cdf)

    return cdf, cdf_corr


def pval_from_cb(cb_min, cb_max, confidence=0.95, distrib='norm'):
    """p-values from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound
    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound
    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters
    pval_corr : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters corrected for
        multiple testing
    """

    n_features = cb_min.size

    if distrib == 'norm':
        quantile = norm.ppf(1 - (1 - confidence) / 2)

    beta_hat = (cb_min + cb_max) / 2

    zscore = np.abs(beta_hat) / (cb_max - cb_min) * 2 * quantile

    if distrib == 'norm':
        pval = 2 * norm.sf(zscore)  # pval = 2 * (1 - norm.cdf(zscore))

    pval_corr = np.minimum(1, pval * n_features)

    return pval, pval_corr


def zscore_from_cb(cb_min, cb_max, confidence=0.95, distrib='norm'):
    """z-scores from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound
    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound
    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-score values
    """

    if distrib == 'norm':
        quantile = norm.ppf(1 - (1 - confidence) / 2)

    beta_hat = (cb_min + cb_max) / 2

    zscore = beta_hat / (cb_max - cb_min) * 2 * quantile

    return zscore


def zscore_from_sf(sf, distrib='norm'):
    """z-scores from survival function values.

    Parameters
    -----------
    sf : ndarray, shape (n_features,)
        Survival function values
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-score values
    """
    if distrib == 'norm':
        zscore = norm.isf(sf)

    return zscore


def zscore_from_cdf(cdf, distrib='norm'):
    """z-scores from cumulative distribution function values.

    Parameters
    -----------
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-score values
    """
    if distrib == 'norm':
        zscore = norm.ppf(cdf)

    return zscore


def zscore_from_sf_and_cdf(sf, cdf, distrib='norm'):
    """z-scores from survival function and cumulative distribution function
    values.

    Parameters
    -----------
    sf : ndarray, shape (n_features,)
        Survival function values
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-score values
    """
    if distrib == 'norm':
        zscore_sf = zscore_from_sf(sf)
        zscore_cdf = zscore_from_cdf(cdf)

    zscore = np.zeros(sf.size)
    zscore[sf < 0.5] = zscore_sf[sf < 0.5]
    zscore[sf > 0.5] = zscore_cdf[sf > 0.5]

    zscore = _replace_infinity(zscore, replace_val=40, method='plus-one')

    return zscore


def sf_from_pval_and_sign(pval, sign, eps=1e-14):
    """Survival function values from p-value and parameter sign.

    Parameters
    ----------
    pval : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters
    sign : ndarray, shape (n_features,)
        Estimated signs for the parameters
    eps : float, optional
        The machine-precision regularization in the computation of the
        survival function value

    Returns
    -------
    sf : ndarray, shape (n_features,)
        Survival function values
    """

    n_features = pval.size
    sf = 0.5 * np.ones(n_features)

    sf[sign > 0] = pval[sign > 0] / 2
    sf[sign < 0] = 1 - pval[sign < 0] / 2
    sf[sf > 1 - eps] = 1 - eps

    return sf


def cdf_from_pval_and_sign(pval, sign, eps=1e-14):
    """Cumulative distribution function values from p-value and parameter sign.

    Parameters
    ----------
    pval : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters
    sign : ndarray, shape (n_features,)
        Estimated signs for the parameters
    eps : float, optional
        The machine-precision regularization in the computation of the
        cumulative distribution function value

    Returns
    -------
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values
    """

    n_features = pval.size
    cdf = 0.5 * np.ones(n_features)

    cdf[sign > 0] = 1 - pval[sign > 0] / 2
    cdf[sign < 0] = pval[sign < 0] / 2
    cdf[cdf > 1 - eps] = 1 - eps

    return cdf


def sf_and_cdf_from_pval_and_sign(pval, sign, eps=1e-14):
    """Survival and cumulative distribution function values
    from p-value and parameter sign.

    Parameters
    ----------
    pval : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters
    sign : ndarray, shape (n_features,)
        Estimated signs for the parameters
    eps : float, optional
        The machine-precision regularization in the computation of the
        survival or cumulative distribution function value

    Returns
    -------
    sf : ndarray, shape (n_features,)
        Survival function values
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values
    """

    sf = sf_from_pval_and_sign(pval, sign, eps=eps)
    cdf = cdf_from_pval_and_sign(pval, sign, eps=eps)
    sf_corr = sf_corr_from_sf(sf)
    cdf_corr = cdf_corr_from_cdf(cdf)

    return sf, sf_corr, cdf, cdf_corr


def pval_from_zscore(zscore, distrib='norm'):
    """p-values from z-scores.

    Parameters
    ----------
    zscore : ndarray, shape (n_features,)
        z-score values
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters
    pval_corr : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters corrected for
        multiple testing
    """
    n_features = zscore.size

    if distrib == 'norm':
        pval = 2 * norm.sf(np.abs(zscore))

    pval_corr = np.minimum(1, pval * n_features)

    return pval, pval_corr


def pval_from_sf(sf, distrib='norm'):
    """z-scores from survival function values

    Parameters
    ----------
    sf : ndarray, shape (n_features,)
        Survival function values
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters
    pval_corr : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters corrected for
        multiple testing
    """
    pval, pval_corr = pval_from_zscore(zscore_from_sf(sf, distrib=distrib),
                                       distrib=distrib)

    return pval, pval_corr


def pval_from_cdf(cdf, distrib='norm'):
    """z-scores from survival function values

    Parameters
    ----------
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters
    pval_corr : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters corrected for
        multiple testing
    """
    pval, pval_corr = pval_from_zscore(zscore_from_cdf(cdf, distrib=distrib),
                                       distrib=distrib)

    return pval, pval_corr


def pval_from_sf_and_cdf(sf, cdf, distrib='norm'):
    """z-scores from survival function values

    Parameters
    ----------
    sf : ndarray, shape (n_features,)
        Survival function values
    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values
    distrib : str, opitonal (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters
    pval_corr : ndarray, shape (n_features,)
        Estimated (two-sided) p-values of the parameters corrected for
        multiple testing
    """
    pval, pval_corr = \
        pval_from_zscore(zscore_from_sf_and_cdf(sf, cdf, distrib=distrib),
                         distrib=distrib)

    return pval, pval_corr
