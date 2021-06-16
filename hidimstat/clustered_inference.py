import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory

from .stat_tools import sf_from_cb, cdf_from_cb
from .desparsified_lasso import desparsified_lasso, desparsified_group_lasso


def _subsampling(n_samples, train_size, groups=None, seed=0):
    """Random subsampling: computes a list of indices"""

    if groups is None:

        n_subsamples = int(n_samples * train_size)
        train_index = resample(np.arange(n_samples), n_samples=n_subsamples,
                               replace=False, random_state=seed)

    else:

        unique_groups = np.unique(groups)
        n_groups = unique_groups.size
        n_subsample_groups = int(n_groups * train_size)
        train_group = resample(unique_groups, n_samples=n_subsample_groups,
                               replace=False, random_state=seed)
        train_index = np.arange(n_samples)[np.isin(groups, train_group)]

    return train_index


def _ward_clustering(X_init, ward, train_index):
    """Ward clustering applied to full X but computed from a subsample of X"""

    ward = ward.fit(X_init[train_index, :])
    X_reduced = ward.transform(X_init)

    return X_reduced, ward


def _hd_inference(X, y, method, n_jobs=1, memory=None, verbose=0, **kwargs):
    """Wrap-up high-dimensional inference procedures"""

    if method == 'desparsified-lasso':

        beta_hat, cb_min, cb_max = \
            desparsified_lasso(X, y, n_jobs=n_jobs, memory=memory,
                               verbose=verbose, **kwargs)
        sf, sf_corr = sf_from_cb(cb_min, cb_max)
        cdf, cdf_corr = cdf_from_cb(cb_min, cb_max)

    elif method == 'desparsified-group-lasso':

        beta_hat, sf, sf_corr, cdf, cdf_corr = \
            desparsified_group_lasso(X, y, n_jobs=n_jobs, memory=memory,
                                     verbose=verbose, **kwargs)

    else:

        raise ValueError('Unknow method')

    return beta_hat, sf, sf_corr, cdf, cdf_corr


def _degrouping(ward, beta_hat, sf, sf_corr, cdf, cdf_corr):
    """Assigning cluster-wise stats to features contained in the corresponding
    cluster and rescaling estimated parameter"""

    sf_degrouped = ward.inverse_transform(sf)
    sf_corr_degrouped = ward.inverse_transform(sf_corr)
    cdf_degrouped = ward.inverse_transform(cdf)
    cdf_corr_degrouped = ward.inverse_transform(cdf_corr)

    labels = ward.labels_
    clusters_size = np.zeros(labels.size)

    for label in range(labels.max() + 1):
        cluster_size = np.sum(labels == label)
        clusters_size[labels == label] = cluster_size

    if len(beta_hat.shape) == 1:

        beta_hat_degrouped = ward.inverse_transform(beta_hat) / clusters_size

    elif len(beta_hat.shape) == 2:

        n_features = sf_degrouped.shape[0]
        n_times = beta_hat.shape[1]
        beta_hat_degrouped = np.zeros((n_features, n_times))

        for i in range(n_times):

            beta_hat_degrouped[:, i] = \
                ward.inverse_transform(beta_hat[:, i]) / clusters_size

    return (beta_hat_degrouped, sf_degrouped, sf_corr_degrouped,
            cdf_degrouped, cdf_corr_degrouped)


def clustered_inference(X_init, y, ward, n_clusters, train_size=0.3,
                        groups=None, method='desparsified-lasso', seed=0,
                        n_jobs=1, memory=None, verbose=1, **kwargs):
    """Clustered inference algorithm

    Parameters
    -----------
    X_init : ndarray, shape (n_samples, n_features)
        Original data (uncompressed).

    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target.

    ward : sklearn.cluster.FeatureAgglomeration Object
        Scikit-learn object that computes ward hierarchical clustering.

    n_clusters : int
        Number of clusters used for the compression.

    train_size : float, optional (default=0.3)
        Fraction of samples used to compute the clustering.
        If `train_size = 1`, clustering is not random since all the samples
        are used to compute the clustering.

    groups : ndarray, shape (n_samples,), optional (default=None)
        Group labels for every sample. If not None, `groups` is used to build
        the subsamples that serve for computing the clustering.

    method : str, optional (default='desparsified-lasso')
        Method used for making the inference.
        Currently the two methods available are 'desparsified-lasso'
        and 'group-desparsified-lasso'. Use 'desparsified-lasso' for
        non-temporal data and 'group-desparsified-lasso' for temporal data.

    seed: int, optional (default=0)
        Seed used for generating a random subsample of the data.
        This seed controls the clustering randomness.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during parallel steps such as inference.

    memory : str or joblib.Memory object, optional (default=None)
        Used to cache the output of the computation of the clustering
        and the inference. By default, no caching is done. If a string is
        given, it is the path to the caching directory.

    verbose: int, optional (default=1)
        The verbosity level. If `verbose > 0`, we print a message before
        runing the clustered inference.

    **kwargs:
        Arguments passed to the statistical inference function.

    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_times)
        Estimated parameter vector or matrix.

    sf : ndarray, shape (n_features,)
        Survival function values of every feature.

    sf_corr : ndarray, shape (n_features,)
        Corrected survival function values of every feature.

    cdf : ndarray, shape (n_features,)
        Cumulative distribution function values of every feature.

    cdf_corr : ndarray, shape (n_features,)
        Corrected cumulative distribution function values of every feature.

    References
    ----------
    .. [1] Chevalier, J. A., Nguyen, T. B., Thirion, B., & Salmon, J. (2021).
           Spatially relaxed inference on high-dimensional linear models.
           arXiv preprint arXiv:2106.02590.
    """

    memory = check_memory(memory)

    n_samples, n_features = X_init.shape

    if verbose > 0:

        print(f'Clustered inference: n_clusters = {n_clusters}, ' +
              f'inference method = {method}, seed = {seed}')

    # Sampling
    train_index = _subsampling(n_samples, train_size, groups=groups, seed=seed)

    # Clustering
    X, ward = memory.cache(_ward_clustering)(X_init, ward, train_index)

    # Preprocessing
    X = StandardScaler().fit_transform(X)
    y = y - np.mean(y)

    # Inference: computing reduced parameter vector and stats
    beta_hat_red, sf_red, sf_corr_red, cdf_red, cdf_corr_red = \
        _hd_inference(X, y, method, n_jobs=n_jobs, memory=memory, **kwargs)

    # De-grouping
    beta_hat, sf, sf_corr, cdf, cdf_corr = \
        _degrouping(ward, beta_hat_red, sf_red, sf_corr_red, cdf_red,
                    cdf_corr_red)

    return beta_hat, sf, sf_corr, cdf, cdf_corr
