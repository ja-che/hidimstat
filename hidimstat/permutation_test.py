import numpy as np
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.utils import check_random_state, safe_indexing
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from hidimstat.stat_tools import cdf_from_pval_and_sign
from hidimstat.stat_tools import sf_from_pval_and_sign


def permutation_test_cv(X, y, method='SVR', n_permutations=1000, C=None,
                        normalize=False, random_state=None, n_jobs=1,
                        verbose=1):

    if C is None:

        Cs = np.logspace(-7, 1, 9)
        steps = [('SVR', LinearSVR())]
        pipeline = Pipeline(steps)
        parameters = {'SVR__C': Cs}
        grid = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs)
        grid.fit(X, y)
        C = grid.best_params_['SVR__C']
        estimator = LinearSVR(C=C)

    else:

        estimator = LinearSVR(C=C)

    sf_corr, cdf_corr = permutation_test(X, y, estimator,
                                         n_permutations=n_permutations,
                                         normalize=normalize,
                                         random_state=random_state,
                                         n_jobs=n_jobs,
                                         verbose=verbose)

    return sf_corr, cdf_corr


def permutation_test(X, y, estimator, n_permutations=1000,
                     normalize=False, random_state=None, n_jobs=1, verbose=1):

    random_state = check_random_state(random_state)

    stat = _permutation_test_stat(clone(estimator), X, y)

    permutation_stats = \
        Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_permutation_test_stat)(clone(estimator), X,
                                            _shuffle(y, random_state))
            for _ in range(n_permutations))

    permutation_stats = np.array(permutation_stats)
    if normalize:
        permutation_stats = \
            (permutation_stats.T / np.linalg.norm(permutation_stats, axis=1)).T

    pval_corr = _step_down_max_T(stat, permutation_stats)

    stat_sign = np.sign(stat)

    sf_corr = sf_from_pval_and_sign(pval_corr, stat_sign)
    cdf_corr = cdf_from_pval_and_sign(pval_corr, stat_sign)

    return sf_corr, cdf_corr


def _permutation_test_stat(estimator, X, y):

    stat = estimator.fit(X, y).coef_
    return stat


def _shuffle(y, random_state):
    indices = random_state.permutation(len(y))
    return safe_indexing(y, indices)


def _step_down_max_T(stat, permutation_stats):

    n_permutations, n_features = np.shape(permutation_stats)

    index_ordered = np.argsort(np.abs(stat))
    stat_ranked = np.empty(n_features)
    stat_ranked[index_ordered] = np.arange(n_features)
    stat_ranked = stat_ranked.astype(int)
    stat_sorted = np.copy(np.abs(stat)[index_ordered])
    permutation_stats_ordered = \
        np.copy(np.abs(permutation_stats)[:, index_ordered])

    for i in range(1, n_features):
        permutation_stats_ordered[:, i] = \
            np.maximum(permutation_stats_ordered[:, i - 1],
                       permutation_stats_ordered[:, i])

    pval_corr = \
        (np.sum(np.less_equal(stat_sorted, permutation_stats_ordered), axis=0)
         / n_permutations)

    for i in range(n_features - 1)[::-1]:
        pval_corr[i] = np.maximum(pval_corr[i], pval_corr[i + 1])

    pval_corr = np.copy(pval_corr[stat_ranked])

    return pval_corr
