# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>
"""Implementation of Model-X knockoffs inference procedure, introduced in
Candes et al. (2016) " Panning for Gold: Model-X Knockoffs for High-dimensional
Controlled Variable Selection" <https://arxiv.org/abs/1610.02351>; and its
aggregated version for more stable inference results in Nguyen et al. (2020)
`Aggregation of Multiple Knockoffs <https://arxiv.org/abs/2002.09269

"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_memory

from .gaussian_knockoff import (_estimate_distribution,
                                gaussian_knockoff_generation)
from .stat_coef_diff import _coef_diff_threshold, stat_coef_diff
from .utils import fdr_threshold, quantile_aggregation


def model_x_knockoff(X, y, fdr=0.1, offset=1, method='equi',
                     statistics='lasso_cv', shrink=False, centered=True,
                     cov_estimator='ledoit_wolf', verbose=False, memory=None,
                     n_jobs=1, seed=None):
    """Model-X Knockoff inference procedure to control False Discoveries Rate,
    based on Candes et. al. (2016) <https://arxiv.org/abs/1610.02351>

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        design matrix

    y : 1D ndarray (n_samples, )
        response vector

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset to calculate knockoff threshold, offset = 1 is equivalent to
        knockoff+

    method : str, optional
        knockoff construction methods, either equi for equi-correlated knockoff
        or sdp for optimization scheme

    statistics : str, optional
        method to calculate knockoff test score

    shrink : bool, optional
        whether to shrink the empirical covariance matrix

    centered : bool, optional
        whether to standardize the data before doing the inference procedure

    cov_estimator : str, optional
        method of empirical covariance matrix estimation

    seed : int or None, optional
        random seed used to generate Gaussian knockoff variable

    Returns
    -------
    selected : 1D array, int
        vector of index of selected variables

    test_score : 1D array, (n_features, )
        vector of test statistic

    thres : float
        knockoff threshold

    X_tilde : 2D array, (n_samples, n_features)
        knockoff design matrix
    """
    memory = check_memory(memory)

    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(
        X, shrink=shrink, cov_estimator=cov_estimator)

    X_tilde = gaussian_knockoff_generation(X, mu, Sigma, memory=memory,
                                           method=method, seed=seed)
    test_score = memory.cache(
        stat_coef_diff, ignore=['n_jobs', 'joblib_verbose'])(
        X, X_tilde, y, method=statistics, n_jobs=n_jobs)
    thres = _coef_diff_threshold(test_score, fdr=fdr, offset=offset)

    selected = np.where(test_score >= thres)[0]

    if verbose:
        return selected, test_score, thres, X_tilde

    return selected


def knockoff_aggregation(X, y, fdr=0.1, offset=1, fdr_control='bhq',
                         n_bootstraps=25, centered=True, shrink=False,
                         construct_method='equi', reshaping_function=None,
                         statistic='lasso_cv', cov_estimator='ledoit_wolf',
                         joblib_verbose=0, n_jobs=1,
                         adaptive_aggregation=False, gamma=0.5, gamma_min=0.05,
                         verbose=False, memory=None, random_state=None):
    """Aggregation of Multiple Knockoffs to control False Discoveries Rate,
    based on Nguyen et. al. (2020) <https://arxiv.org/abs/2002.09269>

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        design matrix

    y : 1D ndarray (n_samples, )
        response vector

    fdr : float, optional
        desired controlled FDR level

    fdr_control : str, optional
        method for controlling FDR

    offset : int, 0 or 1, optional
        offset to calculate knockoff threshold, offset = 1 is equivalent to
        knockoff+

    n_bootstraps : int, optional
        number of knockoff bootstraps

    construct_method : str, optional
        knockoff construction methods, either equi for equi-correlated knockoff
        or sdp for optimization scheme

    statistics : str, optional
        method to calculate knockoff test score

    adaptive_aggregation : bool, optional
        whether to use adaptive quantile aggregation scheme for bootstrapping 
        p-values, for more info see Meinhausen et al. (2009)
        <https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.tm08647>

    shrink : bool, optional
        whether to shrink the empirical covariance matrix

    centered : bool, optional
        whether to standardize the data before doing the inference procedure

    cov_estimator : str, optional
        method of empirical covariance matrix estimation

    n_jobs : int, optional
        number of parallel jobs to run, increase this number will make the
        inference faster, but take more computational resource

    random_state : int or None, optional
        random seed used to generate Gaussian knockoff variable

    Returns
    -------
    selected : 1D array, int
        vector of index of selected variables

    aggregated_pval : 1D array, float
        vector of aggregated pvalues

    pvals : 2D array, float, (n_bootstraps, n_features)
        list of bootrapping pvalues

    """
    # unnecessary to have n_jobs > number of bootstraps
    n_jobs = min(n_bootstraps, n_jobs)

    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(
        X, shrink=shrink, cov_estimator=cov_estimator)

    mem = check_memory(memory)
    stat_coef_diff_cached = mem.cache(stat_coef_diff,
                                      ignore=['n_jobs', 'joblib_verbose'])

    if n_bootstraps == 1:
        X_tilde = gaussian_knockoff_generation(
            X, mu, Sigma, method=construct_method,
            memory=memory, seed=random_state)
        ko_stat = stat_coef_diff_cached(X, X_tilde, y, method=statistic)
        pvals = _empirical_pval(ko_stat, offset)
        threshold = fdr_threshold(pvals, fdr=fdr,
                                  method=fdr_control)
        selected = np.where(pvals <= threshold)[0]

        if verbose:
            return selected, pvals

        return selected

    if isinstance(random_state, (int, np.int32, np.int64)):
        rng = check_random_state(random_state)
    elif random_state is None:
        rng = check_random_state(0)
    else:
        raise TypeError('Wrong type for random_state')

    seed_list = rng.randint(1, np.iinfo(np.int32).max, n_bootstraps)
    parallel = Parallel(n_jobs, verbose=joblib_verbose)
    X_tildes = parallel(delayed(gaussian_knockoff_generation)(
        X, mu, Sigma, method=construct_method, memory=memory,
        seed=seed) for seed in seed_list)

    ko_stats = parallel(delayed(stat_coef_diff_cached)(
        X, X_tildes[i], y, method=statistic) for i in range(n_bootstraps))

    pvals = np.array([_empirical_pval(ko_stats[i], offset)
                      for i in range(n_bootstraps)])

    aggregated_pval = quantile_aggregation(
        pvals, gamma=gamma, gamma_min=gamma_min,
        adaptive=adaptive_aggregation)

    threshold = fdr_threshold(aggregated_pval, fdr=fdr, method=fdr_control,
                              reshaping_function=reshaping_function)
    selected = np.where(aggregated_pval <= threshold)[0]

    if verbose:
        return selected, aggregated_pval, pvals

    return selected


def _empirical_pval(test_score, offset=1):
    """Function to convert knockoff test statistics to empirical pvalues. More
    info in Nguyen et al. (2020) <https://arxiv.org/abs/2002.09269>

    """
    pvals = []
    n_features = test_score.size

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    test_score_inv = -test_score
    for i in range(n_features):
        if test_score[i] <= 0:
            pvals.append(1)
        else:
            pvals.append((offset +
                np.sum(test_score_inv >= test_score[i])) / n_features)

    return np.array(pvals)
