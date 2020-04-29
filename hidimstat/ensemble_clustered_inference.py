import numpy as np
from joblib import Parallel, delayed

from .multi_sample_split import aggregate_medians, aggregate_quantiles
from .clustered_inference import clustered_inference


def ensemble_clustered_inference(X_init, y, ward, n_clusters, method='DL',
                                 aggregate='quantiles', gamma_min=0.2,
                                 train_size=0.7, condition_mask=None,
                                 groups=None, seed=0, n_rand=25, predict=False,
                                 n_jobs=1):
    """EnCluDL"""

    results = Parallel(n_jobs=n_jobs)(
        delayed(clustered_inference)(X_init, y, ward, n_clusters, method,
                                     train_size, condition_mask, groups, rand,
                                     predict)
        for rand in np.arange(seed, seed + n_rand))

    results = np.asarray(results)

    list_sf = results[:, 0, :]
    list_sf_corr = results[:, 1, :]
    list_cdf = results[:, 2, :]
    list_cdf_corr = results[:, 3, :]

    if aggregate == 'quantiles':

        sf = aggregate_quantiles(list_sf, gamma_min)
        sf_corr = aggregate_quantiles(list_sf_corr, gamma_min)
        cdf = aggregate_quantiles(list_cdf, gamma_min)
        cdf_corr = aggregate_quantiles(list_cdf_corr, gamma_min)

    elif aggregate == 'medians':

        sf = aggregate_medians(list_sf)
        sf_corr = aggregate_medians(list_sf_corr)
        cdf = aggregate_medians(list_cdf)
        cdf_corr = aggregate_medians(list_cdf_corr)

    if predict:

        list_beta_hat = results[:, 4, :]
        beta_hat = np.mean(np.asarray(list_beta_hat), axis=0)

        return sf, sf_corr, cdf, cdf_corr, beta_hat

    return sf, sf_corr, cdf, cdf_corr
