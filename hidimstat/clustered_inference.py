import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

from .stat_tools import sf_from_cb, cdf_from_cb
from .desparsified_lasso import desparsified_lasso


def clustered_inference(X_init, y, ward, n_clusters, method='DL',
                        train_size=0.7, condition_mask=None,
                        groups=None, rand=0, predict=False):
    """CluDL"""

    n_trials, n_voxels = X_init.shape

    # Sampling
    if groups is None:

        train_index = resample(np.arange(n_trials),
                               n_samples=int(n_trials * train_size),
                               replace=False, random_state=rand)

    else:

        unique_groups = np.unique(groups)
        n_groups = unique_groups.size
        train_group = resample(unique_groups,
                               n_samples=int(n_groups * train_size),
                               replace=False, random_state=rand)
        train_index = np.arange(n_trials)[np.isin(groups, train_group)]

    ward.fit(X_init[train_index, :])
    X_reduced = ward.transform(X_init)

    if condition_mask is None:
        X = np.asarray(X_reduced)

    else:
        X = np.asarray(X_reduced[condition_mask])

    # Experiment
    X = StandardScaler().fit_transform(X)
    if method != 'UGaonkar':
        y = y - np.mean(y)

    if predict:
        sf, sf_corr, cdf, cdf_corr, beta_hat = hd_inference(X, y, method,
                                                            predict=True)
    else:
        sf, sf_corr, cdf, cdf_corr = hd_inference(X, y, method)

    sf_compressed = ward.inverse_transform(sf)
    sf_corr_compressed = ward.inverse_transform(sf_corr)
    cdf_compressed = ward.inverse_transform(cdf)
    cdf_corr_compressed = ward.inverse_transform(cdf_corr)

    if predict:

        labels = ward.labels_
        clusters_size = np.zeros(labels.size)

        for label in range(labels.max() + 1):
            cluster_size = np.sum(labels == label)
            clusters_size[labels == label] = cluster_size

        beta_hat_compressed = \
            ward.inverse_transform(beta_hat) / clusters_size

        return (sf_compressed, sf_corr_compressed, cdf_compressed,
                cdf_corr_compressed, beta_hat_compressed)

    return (sf_compressed, sf_corr_compressed, cdf_compressed,
            cdf_corr_compressed)


def hd_inference(X, y, method='DL', predict=False, n_jobs=1):

    if method == 'DL':

        beta_hat, cb_min, cb_max = desparsified_lasso(X, y, n_jobs=n_jobs)
        sf, sf_corr = sf_from_cb(cb_min, cb_max)
        cdf, cdf_corr = cdf_from_cb(cb_min, cb_max)

    else:
        raise ValueError('Unknow method')

    if predict:
        return sf, sf_corr, cdf, cdf_corr, beta_hat

    return sf, sf_corr, cdf, cdf_corr
