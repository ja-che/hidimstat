import numpy as np


def aggregate_medians(list_sf):
    """Aggregation of survival function values with the twice median procedure

    Parameters
    -----------
        list_sf : ndarray or scipy.sparse matrix, (n_iter, n_features)
            List of survival fuction values
    """

    n_iter, n_features = list_sf.shape

    sf = np.median(list_sf, axis=0)
    sf[sf > 0.5] = np.maximum(0.5, 1 - (1 - sf[sf > 0.5]) * 2)
    sf[sf < 0.5] = np.minimum(0.5, sf[sf < 0.5] * 2)

    return sf


def aggregate_quantiles(list_sf, gamma_min=0.2):
    """Aggregation of survival function values with the Meinshausen procedure

    Parameters
    -----------
        list_sf : ndarray or scipy.sparse matrix, (n_iter, n_features)
            List of survival fuction values
    """

    n_iter, n_features = list_sf.shape
    sf = 0.5 * np.ones(n_features)

    m = n_iter + 1
    k = np.maximum(1, int(np.floor(gamma_min * n_iter)))
    r = 1 - np.log(gamma_min)
    seq = range(k, n_iter)

    asc_sf = np.sort(list_sf, axis=0)
    dsc_sf = asc_sf[::-1]

    for i in np.arange(n_features):

        sf_neg = min([asc_sf[j, i] * m / (j + 1) for j in seq])
        sf_neg = min(0.5, sf_neg)

        sf_pos = max([1 - (1 - dsc_sf[j, i]) * m / (j + 1) for j in seq])
        sf_pos = max(0.5, sf_pos)

        if (1 - sf_pos) < sf_neg:

            sf[i] = np.maximum(0.5, 1 - (1 - sf_pos) * r)

        else:

            sf[i] = np.minimum(0.5, sf_neg * r)

    return sf
