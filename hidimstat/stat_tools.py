import numpy as np
from scipy.stats import norm


def sf_corr_from_sf(sf):

    n_features = sf.size

    sf_corr = np.zeros(n_features) + 0.5

    sf_corr[sf < 0.5] = np.minimum(0.5, sf[sf < 0.5] * n_features)
    sf_corr[sf > 0.5] = np.maximum(0.5, 1 - (1 - sf[sf > 0.5]) * n_features)

    return sf_corr


def cdf_corr_from_cdf(cdf):

    n_features = cdf.size

    cdf_corr = np.zeros(n_features) + 0.5

    cdf_corr[cdf < 0.5] = np.minimum(0.5, cdf[cdf < 0.5] * n_features)
    cdf_corr[cdf > 0.5] = \
        np.maximum(0.5, 1 - (1 - cdf[cdf > 0.5]) * n_features)

    return cdf_corr


def sf_from_cb(cb_min, cb_max, confidence=0.95, distrib='Norm', eps=1e-14):
    """Survival function values from confidence intervals

    Parameters
    -----------
        cb_min : float
            Value of the inferior confidence bound
        cb_max : float
            Value of the superior confidence bound
        confidence : float, optional
            Confidence level used to compute the confidence intervals.
            Each value should be in the range [0, 1].
        eps : float, optional
            The machine-precision regularization in the computation of the
            survival function value
    """

    if distrib == 'Norm':
        quantile = norm.ppf(1 - (1 - confidence) / 2)

    beta_hat = (cb_min + cb_max) / 2

    t_stat = beta_hat / (cb_max - cb_min) * 2 * quantile

    if distrib == 'Norm':
        sf = norm.sf(t_stat)

    sf[sf > 1 - eps] = 1 - eps
    sf_corr = sf_corr_from_sf(sf)

    return sf, sf_corr


def cdf_from_cb(cb_min, cb_max, confidence=0.95, distrib='Norm', eps=1e-14):
    """Cumulative distribution function values from confidence intervals

    Parameters
    -----------
        cb_min : float
            Value of the inferior confidence bound
        cb_max : float
            Value of the superior confidence bound
        confidence : float, optional
            Confidence level used to compute the confidence intervals.
            Each value should be in the range [0, 1].
        eps : float, optional
            The machine-precision regularization in the computation of the
            survival function value
    """

    if distrib == 'Norm':
        quantile = norm.ppf(1 - (1 - confidence) / 2)

    beta_hat = (cb_min + cb_max) / 2

    t_stat = beta_hat / (cb_max - cb_min) * 2 * quantile

    if distrib == 'Norm':
        cdf = norm.cdf(t_stat)

    cdf[cdf > 1 - eps] = 1 - eps
    cdf_corr = cdf_corr_from_cdf(cdf)

    return cdf, cdf_corr
